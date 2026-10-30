"""Accuracy, GPU time and executed FLOPs of every SVD in Pavel-s_SVD across matrix types.

Same solvers as benchmarks/gpu_flops (17 standalone algorithms behind svdlab.blocks.svd, four
svdlab.recipes and torch.linalg.svd as reference), run on square n x n matrices of different
structure: Gaussian, ill-conditioned, low-rank, rank-deficient, clustered singular values,
symmetric and complex channel-like. One record per (matrix type, n, method, dtype).

Run from the repository root:
    .venv/bin/python benchmarks/matrix_types/bench_matrix_types.py --sizes 32 64 128 256 --device cuda
Quick check:
    .venv/bin/python benchmarks/matrix_types/bench_matrix_types.py --sizes 32 --repeat 2 --types gaussian symmetric
"""
import argparse, json, math, os, statistics, sys, time, traceback
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'gpu_flops'))
import bench_svd as B   # make_impls() and the iteration counters on the primitives
import flops_svd as F   # element-wise FLOP models wrapped around the same primitives

OUT = 'benchmarks/matrix_types/out'
SIZES = [32, 64, 128, 256]
WARMUP, REPS = 1, 5
SLOW_RUN_S = 2.0        # a single run slower than this -> 1 warm-up + 2 timed runs
SKIP_AFTER_128_S = 5.0  # median at n=128 above this -> skip n=256 for that (type, method, dtype)
F64 = ['torch.linalg.svd', 'blocks.svd:jacobi', 'blocks.svd:bidiag_dc', 'blocks.svd:polar_qdwh',
       'blocks.svd:gram_jacobi']


# --- matrix types ---------------------------------------------------------------------
# Every generator returns a float64 or complex128 CPU tensor; it is cast to the working dtype
# afterwards, and the reference singular values are taken from the cast matrix in float64.

def _haar(n, g, cplx=False):
    z = torch.randn(n, n, generator=g, dtype=torch.float64)
    if cplx:
        z = torch.complex(z, torch.randn(n, n, generator=g, dtype=torch.float64))
    q, r = torch.linalg.qr(z)
    d = torch.diagonal(r)
    return q * (d / d.abs())


def _with_sigmas(s, g):
    n = s.shape[0]
    return (_haar(n, g) * s) @ _haar(n, g).mT


def gaussian(n, g):
    return torch.randn(n, n, generator=g, dtype=torch.float64)


def cond(k):
    def f(n, g):
        return _with_sigmas(torch.logspace(0, -k, n, dtype=torch.float64), g)
    f.__doc__ = f'U diag(logspace(0, -{k}, n)) V^T with Haar U, V: condition number 1e{k}, graded spectrum.'
    return f


def low_rank(n, g):
    r = max(4, n // 16)
    s = torch.cat([torch.logspace(0, -1, r, dtype=torch.float64),
                   torch.logspace(-5, -6, n - r, dtype=torch.float64)])
    return _with_sigmas(s, g)


def rank_deficient(n, g):
    r = n // 2
    s = torch.cat([torch.logspace(0, -2, r, dtype=torch.float64), torch.zeros(n - r, dtype=torch.float64)])
    return _with_sigmas(s, g)


def clustered(n, g):
    centers = torch.tensor([1.0, 0.5, 0.1, 0.01], dtype=torch.float64)
    idx = torch.arange(n) * len(centers) // n
    jitter = 1e-6 * torch.rand(n, generator=g, dtype=torch.float64)
    return _with_sigmas(torch.sort(centers[idx] * (1 + jitter), descending=True).values, g)


def symmetric(n, g):
    z = torch.randn(n, n, generator=g, dtype=torch.float64)
    return (z + z.mT) / math.sqrt(2)


def channel_iid(n, g):
    re = torch.randn(n, n, generator=g, dtype=torch.float64)
    im = torch.randn(n, n, generator=g, dtype=torch.float64)
    return torch.complex(re, im) / math.sqrt(2)


def channel_corr(n, g, rho=0.9):
    i = torch.arange(n, dtype=torch.float64)
    r = rho ** (i[:, None] - i[None, :]).abs()
    w, q = torch.linalg.eigh(r)
    half = ((q * w.clamp_min(0).sqrt()) @ q.mT).to(torch.complex128)
    return half @ channel_iid(n, g) @ half


gaussian.__doc__ = 'i.i.d. N(0, 1) entries (the baseline of benchmarks/gpu_flops).'
low_rank.__doc__ = 'Numerically low rank: n/16 singular values in [0.1, 1], the rest in [1e-6, 1e-5].'
rank_deficient.__doc__ = 'Exactly rank n/2: singular values logspace(0, -2), then exact zeros (before the cast).'
clustered.__doc__ = 'Four clusters of n/4 singular values at 1, 0.5, 0.1, 0.01, relative spread 1e-6 (near-repeated).'
symmetric.__doc__ = '(G + G^T)/sqrt(2): real symmetric indefinite, singular values = |eigenvalues|.'
channel_iid.__doc__ = 'Complex Rayleigh MIMO channel: i.i.d. CN(0, 1) entries.'
channel_corr.__doc__ = 'Complex Kronecker MIMO channel R^1/2 G R^1/2, exponential correlation rho = 0.9 at both ends.'

TYPES = {'gaussian': gaussian, 'cond_1e3': cond(3), 'cond_1e6': cond(6), 'low_rank': low_rank,
         'rank_deficient': rank_deficient, 'clustered': clustered, 'symmetric': symmetric,
         'channel_iid': channel_iid, 'channel_corr': channel_corr}


def matrix(kind, n, dtype, dev):
    g = torch.Generator().manual_seed(1000 + n + 100003 * list(TYPES).index(kind))
    a = TYPES[kind](n, g)
    if a.is_complex():
        dtype = {torch.float32: torch.complex64, torch.float64: torch.complex128}[dtype]
    return a.to(dtype).to(dev)


# --- measurements ---------------------------------------------------------------------

def accuracy(a, u, s, vh, s_ref):
    """Errors computed in float64 / complex128 from the returned factors."""
    wide = torch.complex128 if a.is_complex() else torch.float64
    a64, u, vh = a.to(wide), u.to(wide), vh.to(wide)
    s = s.real.double() if s.is_complex() else s.double()
    k = s.shape[0]
    rec = torch.linalg.norm(a64 - (u * s.to(wide)) @ vh) / torch.linalg.norm(a64)
    eye = torch.eye(k, dtype=wide, device=a.device)
    ou = torch.linalg.norm(u.mH @ u - eye)
    ov = torch.linalg.norm(vh @ vh.mH - eye)
    sr = s_ref[:k]
    diff = (s - sr).abs()
    eps = torch.finfo(torch.float32 if a.dtype in (torch.float32, torch.complex64) else torch.float64).eps
    resolvable = sr > a.shape[-1] * eps * sr[0]
    rel = (diff[resolvable] / sr[resolvable]).max() if bool(resolvable.any()) else torch.zeros(())
    return dict(recon_err=float(rec), orth_u=float(ou), orth_v=float(ov),
                sv_err_norm=float(diff.max() / sr[0]), sv_err_rel=float(rel),
                n_resolvable=int(resolvable.sum()))


def timed(fn, a, n, warmup, reps):
    cuda = a.device.type == 'cuda'
    times, out = [], None
    i = 0
    while i < warmup + reps:
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            out = fn(a, n)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
        else:
            t = time.perf_counter()
            out = fn(a, n)
            ms = (time.perf_counter() - t) * 1e3
        if i >= warmup:
            times.append(ms)
        if i == 0 and ms > SLOW_RUN_S * 1000:
            warmup, reps = 1, min(reps, 2)
        i += 1
    return out, times


def count_flops(fn, a, n):
    """matmul FLOPs (FlopCounterMode) + exact element-wise FLOPs of the primitives, one solve.
    Complex inputs: multiplied by 4 to give real-arithmetic equivalents (approximate)."""
    F.PRIM.clear()
    with FlopCounterMode(display=False) as fc:
        fn(a, n)
    mm, prim = int(fc.get_total_flops()), dict(F.PRIM)
    mult = 4 if a.is_complex() else 1
    return dict(matmul_flops=mm * mult, elementwise_flops=sum(prim.values()) * mult,
                total_flops=(mm + sum(prim.values())) * mult, primitive_flops=prim)


def run(kind, name, fn, n, dtype, s_ref, dev, warmup, reps):
    a = matrix(kind, n, dtype, dev)
    rec = dict(type=kind, impl=name, n=n, dtype=str(a.dtype).replace('torch.', ''), status='ok')
    try:
        B.COUNTS.clear()
        u, s, vh, info = fn(a, n)          # first call: iteration counts and solver stats
        rec['iters'] = {k: (v[0] if len(v) == 1 else dict(calls=len(v), total=sum(v), max=max(v)))
                        for k, v in B.COUNTS.items()} or None
        rec['stats'] = {k: v for k, v in info.items() if k != 'algorithm'} or None
        (u, s, vh, _), times = timed(fn, a, n, warmup, reps)
        rec.update(median_ms=statistics.median(times), min_ms=min(times), timed_runs=len(times))
        rec.update(accuracy(a, u, s, vh, s_ref))
        if s.shape[0] != n:
            rec['status'] = f'ok (returned rank {s.shape[0]})'
        if name == 'torch.linalg.svd':
            rec.update(total_flops=21 * n ** 3 * (4 if a.is_complex() else 1), flops_estimated=True)
        else:
            rec.update(count_flops(fn, a, n))
    except Exception as e:
        rec['status'] = 'failed'
        rec['error'] = f'{type(e).__name__}: {e}'.strip()[:500]
        rec['traceback_tail'] = traceback.format_exc().strip().splitlines()[-4:]
    if dev.startswith('cuda'):
        torch.cuda.synchronize()
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    p.add_argument('--types', nargs='+', default=list(TYPES), choices=list(TYPES))
    p.add_argument('--impls', nargs='*', help='substring filter on method names')
    p.add_argument('--device', default='cuda', help='cuda, cuda:N or cpu (cpu: wall-clock timing)')
    p.add_argument('--warmup', type=int, default=WARMUP)
    p.add_argument('--repeat', type=int, default=REPS, help='timed runs per case')
    p.add_argument('--no-f64', action='store_true', help='skip the float64 runs')
    p.add_argument('--out', default=OUT)
    args = p.parse_args()
    B.DEV = args.device
    os.makedirs(args.out, exist_ok=True)
    path = f'{args.out}/results.json'
    results = json.load(open(path)) if os.path.exists(path) else []   # resume
    done = {(r['type'], r['impl'], r['n'], r['dtype'][-2:]) for r in results}
    impls = B.make_impls()
    if args.impls:
        impls = {k: v for k, v in impls.items() if any(s in k for s in args.impls)}
    plan = [(k, torch.float32) for k in impls] + ([] if args.no_f64 else [(k, torch.float64) for k in F64 if k in impls])
    skip = {(r['type'], r['impl'], r['dtype'][-2:]) for r in results
            if r['n'] == 128 and (r.get('median_ms') or 0) > SKIP_AFTER_128_S * 1000}
    t0 = time.time()
    for kind in args.types:
        for n in args.sizes:
            s_ref = {}
            for name, dtype in plan:
                bits = str(dtype)[-2:]
                if (kind, name, n, bits) in done:
                    continue
                if n > 128 and (kind, name, bits) in skip:
                    results.append(dict(type=kind, impl=name, n=n, dtype=str(dtype).replace('torch.', ''),
                                        status='skipped', error=f'n=128 median exceeded {SKIP_AFTER_128_S} s'))
                    continue
                if dtype not in s_ref:
                    a = matrix(kind, n, dtype, 'cpu')
                    s_ref[dtype] = torch.linalg.svdvals(a.to(torch.complex128 if a.is_complex() else torch.float64)).to(args.device)
                rec = run(kind, name, impls[name], n, dtype, s_ref[dtype], args.device, args.warmup, args.repeat)
                results.append(rec)
                print(f'[{time.time()-t0:7.1f}s] {kind:14s} n={n:3d} {rec["dtype"]:10s} {name:30s} {rec["status"]:7s} '
                      f'med={rec.get("median_ms")} rec={rec.get("recon_err")} sv={rec.get("sv_err_rel")} '
                      f'{rec.get("error") or ""}', flush=True)
                if n == 128 and (rec.get('median_ms') or 0) > SKIP_AFTER_128_S * 1000:
                    skip.add((kind, name, bits))
                with open(path, 'w') as fh:
                    json.dump(results, fh, indent=1)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version.split()[0],
               gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'cpu',
               sizes=args.sizes, types={k: TYPES[k].__doc__ for k in args.types}, warmup=args.warmup,
               reps=args.repeat, seed='torch.Generator().manual_seed(1000 + n + 100003 * type_index)',
               last_run_s=round(time.time() - t0, 1))
    with open(f'{args.out}/env.json', 'w') as fh:
        json.dump(env, fh, indent=1)
    print(json.dumps(env, indent=1))


if __name__ == '__main__':
    main()
