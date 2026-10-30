"""Standalone dense SVD families: arithmetic only, with explicit output contracts."""
import math
import torch
from . import primitives as P
from .divide_conquer import bidiagonal_dc, tridiagonal_dc

ALGORITHMS = (
    'bidiag_qr', 'bidiag_dc', 'jacobi', 'block_jacobi', 'qr', 'tsqr',
    'gram_jacobi', 'gram_dc', 'polar_qdwh', 'polar_newton_schulz',
    'polar_halley', 'polar_polynomial', 'randomized', 'subspace', 'block_krylov', 'lanczos',
    'polar_cans',
)


def _eye(a, n):
    return torch.eye(n, device=a.device, dtype=a.dtype).expand(a.shape[0], n, n).clone()


def _finish_columns(w, s):
    keep = s > P._eps(w.dtype) * max(w.shape[-2:]) * s[:, :1]
    u = w / torch.where(keep, s, torch.ones_like(s))[:, None, :]
    return P.complete_null_columns(u, keep), torch.where(keep, s, torch.zeros_like(s))


def _jacobi(x, stats, opts):
    u, s, v, sweeps = P.jacobi_svd(x, tol=opts.get('tol'), max_sweeps=opts.get('max_sweeps', 60))
    stats['jacobi_sweeps'] = max(stats.get('jacobi_sweeps', 0), sweeps)
    return u, s, v


def _block_jacobi(a, stats, opts):
    b, m, n = a.shape
    width = opts.get('block_size', 4)
    if width < 1:
        raise ValueError('block_size must be positive')
    groups = [list(range(j, min(j + width, n))) for j in range(0, n, width)]
    if len(groups) == 1:
        return _jacobi(a, stats, opts)
    if len(groups) % 2:
        groups.append([])
    w, v = a.clone(), _eye(a, n)
    tol = opts.get('tol') or 40 * P._eps(a.dtype)
    for sweep in range(opts.get('max_sweeps', 30)):
        for pairs in P._round_robin(len(groups)):
            for i, j in pairs:
                cols = groups[i] + groups[j]
                if len(cols) < 2:
                    continue
                _, _, vv = _jacobi(w[:, :, cols], stats, opts)
                w[:, :, cols] = w[:, :, cols] @ vv
                v[:, :, cols] = v[:, :, cols] @ vv
        g = w.mH @ w
        d = g.diagonal(dim1=-2, dim2=-1).real.clamp_min(0)
        denom = d.sqrt()[:, :, None] * d.sqrt()[:, None, :]
        g.diagonal(dim1=-2, dim2=-1).zero_()
        live = d > (P._eps(a.dtype) * max(m, n)) ** 2 * d.amax(-1, keepdim=True)
        corr = torch.where(live[:, :, None] & live[:, None, :], g.abs() / denom.clamp_min(torch.finfo(d.dtype).tiny), 0)
        if bool(corr.max() <= tol):
            break
    else:
        raise RuntimeError('block Jacobi did not converge')
    s = (w.abs() ** 2).sum(1).sqrt()
    order = s.argsort(-1, descending=True)
    s = s.gather(1, order)
    w = w.gather(2, order[:, None].expand_as(w))
    v = v.gather(2, order[:, None].expand_as(v))
    u, s = _finish_columns(w, s)
    stats['block_sweeps'] = sweep + 1
    return u, s, v


def _tsqr(a, rows, stats):
    b, m, n = a.shape
    rows = max(rows, n)
    if m <= rows:
        return P.qr(a)
    leaves = []
    for start in range(0, m, rows):
        end = min(start + rows, m)
        if end - start < n:
            old_start, _, _ = leaves.pop()
            start = old_start
        q, r = P.qr(a[:, start:end])
        leaves.append((start, q, r))
    stats['tsqr_leaves'] = len(leaves)
    # A binary reduction tree, followed by tree back-transformation.
    nodes = [(q, r) for _, q, r in leaves]
    while len(nodes) > 1:
        nxt = []
        for i in range(0, len(nodes), 2):
            if i + 1 == len(nodes):
                nxt.append(nodes[i]); continue
            q1, r1 = nodes[i]; q2, r2 = nodes[i + 1]
            q, r = P.qr(torch.cat([r1, r2], -2))
            nxt.append((torch.cat([q1 @ q[:, :n], q2 @ q[:, n:]], -2), r))
        nodes = nxt
    return nodes[0]


def _gram(a, stats, opts, dc):
    g = a.mH @ a
    g = (g + g.mH) / 2
    if dc:
        d, e, q = P.tridiag(g)
        results = [tridiagonal_dc(di.double(), ei.double(), stats, opts.get('leaf_size', 4)) for di, ei in zip(d, e)]
        w = torch.stack([z[0] for z in results]).to(d.dtype)
        v = q @ torch.stack([z[1] for z in results]).to(a.dtype)
        stats['merge_precision'] = 'float64 on input device'
    else:
        w, v, it = P.jacobi_eigh(g, max_sweeps=opts.get('max_sweeps', 60))
        stats['eigen_sweeps'] = it
    w, v = w.flip(-1), v.flip(-1)
    # The eigenvalue error is absolute, O(eps*||G||). Do not fabricate small values.
    cut = P._eps(a.dtype) * a.shape[-1] * w[:, :1].clamp_min(0)
    keep = w > cut
    s = torch.where(keep, w.clamp_min(0).sqrt(), 0)
    u = (a @ v) / torch.where(keep, s, torch.ones_like(s))[:, None, :]
    u = P.complete_null_columns(u, keep)
    stats['gram_discarded_values'] = int((~keep).sum())
    return u, s, v


def _polar(a, stats, opts, method):
    n = a.shape[-1]
    norm = (a.abs() ** 2).sum((-2, -1), keepdim=True).sqrt().clamp_min(torch.finfo(a.real.dtype).tiny)
    x = a / norm
    ident = _eye(a, n)
    tol = opts.get('tol') or 10 * P._eps(a.dtype)
    lower = opts.get('lower_bound', P._eps(a.dtype))
    if not 0 < lower <= 1:
        raise ValueError('lower_bound must be in (0,1]')
    max_iter = opts.get('max_iter', 120 if method in ('polar_newton_schulz', 'polar_polynomial', 'polar_cans') else 40)
    converged = False
    for it in range(max_iter):
        g = x.mH @ x
        if method == 'polar_newton_schulz':
            xn = 0.5 * x @ (3 * ident - g)
        elif method == 'polar_polynomial':
            xn = x @ (15 * ident - 10 * g + 3 * (g @ g)) / 8
        elif method == 'polar_cans':
            order = opts.get('cans_order') or 3
            bounds = opts.get('cans_bounds') or (tol, 1+tol)
            A, B = bounds
            C = torch.zeros(order, device=a.device)
            if order == 2:
                C[0] = 2*(A ** 2 + A*B + B ** 2) / \
                    (2 * ((A ** 2 + A*B + B ** 2)/3) ** 1.5 + A**2 * B + A*B**2)
                C[1] = - 2 / (2 * ((A ** 2 + A*B + B ** 2)/3) ** 1.5 + A**2 * B + A*B**2)
                e = (2 * ((A ** 2 + A*B + B ** 2)/3) ** 1.5 - A**2 * B - A*B**2) / \
                    (2 * ((A ** 2 + A*B + B ** 2)/3) ** 1.5 + A**2 * B + A*B**2)
            else:
                c_all,e = remez(A, B, order) 
                C = c_all[1::2]
            opts['cans_bounds'] = (1-e, 1+e)
            xn = C[0] * x
            gg = g
            for i in range(1, C.shape[0]):
                xn += C[i] * x @ gg
                gg = gg @ g
        else:
            if method == 'polar_halley':
                aa, bb, cc = 3., 1., 3.
            else:
                l2 = lower * lower
                dd = (4 * (1 - l2) / (l2 * l2)) ** (1 / 3)
                sd = math.sqrt(1 + dd)
                aa = sd + 0.5 * math.sqrt(max(8 - 4 * dd + 8 * (2 - l2) / (l2 * sd), 0))
                bb = (aa - 1) ** 2 / 4
                cc = aa + bb - 1
                lower = min(1., lower * (aa + bb * l2) / (1 + cc * l2))
            if cc > 100:
                q, _ = P.qr(torch.cat([math.sqrt(cc) * x, ident], -2))
                top, bot = q[:, :a.shape[-2]], q[:, a.shape[-2]:]
                xn = bb / cc * x + (aa - bb / cc) / math.sqrt(cc) * (top @ bot.mH)
            else:
                z = ident + cc * g
                chol, fail = P.cholesky((z + z.mH) / 2)
                if bool(fail.any()):
                    raise RuntimeError('polar Cholesky lost positive definiteness')
                invprod = P.trsm_upper(chol.mH, P.trsm_lower(chol, x.mH)).mH
                xn = bb / cc * x + (aa - bb / cc) * invprod
        diff = (xn - x).abs().amax()
        x = xn
        if bool(diff <= tol * x.abs().amax().clamp_min(1)):
            converged = True
            break
    if not converged:
        raise RuntimeError(f'{method} did not converge in {max_iter} iterations')
    h = x.mH @ a
    h = (h + h.mH) / 2
    w, v, sweeps = P.jacobi_eigh(h, max_sweeps=opts.get('max_sweeps', 60))
    s, v = w.flip(-1).clamp_min(0), v.flip(-1)
    keep = s > P._eps(a.dtype) * max(a.shape[-2:]) * s[:, :1]
    u = P.complete_null_columns(x @ v, keep)
    s = torch.where(keep, s, 0)
    stats.update(polar_iterations=it + 1, eigen_sweeps=sweeps,
                 polar_fixed_point_error=float((x @ (x.mH @ x) - x).abs().amax()))
    return u, s, v


def _sketch(a, columns, seed):
    gen = torch.Generator(device=a.device).manual_seed(seed)
    return torch.randn(a.shape[0], a.shape[-1], columns, dtype=a.dtype, device=a.device, generator=gen)


def _partial(a, stats, opts, method):
    k = opts.get('rank', min(a.shape[-2:]))
    ell = min(k + opts.get('oversample', 8), min(a.shape[-2:]))
    omega = opts.get('initial_basis')
    if omega is None:
        omega = _sketch(a, ell, opts.get('seed', 20261011))
    else:
        if omega.ndim == 2:
            omega = omega[None]
        if omega.shape != (a.shape[0], a.shape[-1], ell):
            raise ValueError('initial_basis shape must match batch, n, rank+oversample')
        omega = omega.to(device=a.device, dtype=a.dtype)
        stats['warm_started'] = True
    q, _ = P.qr(a @ omega)
    bases = [q]
    iterations = opts.get('iterations', 2)
    for _ in range(iterations):
        z, _ = P.qr(a.mH @ q)
        y = a @ z
        if method == 'block_krylov':
            joined = torch.cat(bases, -1)
            available = a.shape[-2] - joined.shape[-1]
            if available <= 0:
                break
            for _ in range(2):
                y = y - joined @ (joined.mH @ y)
            q, _ = P.qr(y[:, :, :min(ell, available)])
            bases.append(q)
        else:
            q, _ = P.qr(y)
    if method == 'block_krylov':
        # Householder QR restores orthogonality at a numerical Krylov breakdown.
        q, _ = P.qr(torch.cat(bases, -1))
    small = q.mH @ a
    u, s, v = _raw(small, 'jacobi', stats, opts)
    stats.update(sketch_columns=ell, range_columns=q.shape[-1], power_iterations=iterations)
    return (q @ u)[:, :, :k], s[:, :k], v[:, :, :k]


def _raw(a, method, stats, opts):
    # All full-spectrum kernels operate on a tall representation.
    if a.shape[-2] < a.shape[-1] and method not in ('randomized', 'subspace', 'block_krylov', 'lanczos'):
        u, s, v = _raw(a.mH, method, stats, opts)
        return v, s, u
    if method == 'jacobi':
        return _jacobi(a, stats, opts)
    if method == 'block_jacobi':
        return _block_jacobi(a, stats, opts)
    if method == 'bidiag_qr':
        d, e, q, p = P.bidiag(a)
        s, ub, vb = P.bdsqr(d, e, max_iter=opts.get('max_iter'), tol=opts.get('tol'))
        stats['qr_iterations'] = P.bdsqr.last_iters
        return q @ ub.to(a.dtype), s, p @ vb.to(a.dtype)
    if method == 'bidiag_dc':
        d, e, q, p = P.bidiag(a)
        bb = torch.diag_embed(d.double()) + torch.diag_embed(e.double(), offset=1)
        out = [bidiagonal_dc(bi, stats, opts.get('leaf_size', 4)) for bi in bb]
        u, s, v = (torch.stack([x[j] for x in out]) for j in range(3))
        stats['merge_precision'] = 'float64 on input device'
        return q @ u.to(a.dtype), s.to(a.real.dtype), p @ v.to(a.dtype)
    if method in ('qr', 'tsqr'):
        q, r = P.qr(a) if method == 'qr' else _tsqr(a, opts.get('tile_rows', 4 * a.shape[-1]), stats)
        u, s, v = _jacobi(r, stats, opts)
        return q @ u, s, v
    if method in ('gram_jacobi', 'gram_dc'):
        return _gram(a, stats, opts, method == 'gram_dc')
    if method.startswith('polar_'):
        return _polar(a, stats, opts, method)
    if method in ('randomized', 'subspace', 'block_krylov'):
        return _partial(a, stats, opts, method)
    if method == 'lanczos':
        from .iterative import lanczos_svd, DenseOperator
        out = [lanczos_svd(DenseOperator(x), rank=opts.get('rank', min(a.shape[-2:])),
                           steps=opts.get('steps'), seed=opts.get('seed', 20261011)) for x in a]
        stats['lanczos_steps'] = max(x[3]['steps'] for x in out)
        return tuple(torch.stack([x[j] for x in out]) for j in range(3))
    raise ValueError(f'unknown standalone algorithm {method!r}')


def svd(a, algorithm='jacobi', *, full_matrices=False, values_only=False,
        return_info=False, **opts):
    """Return U, s, Vh (and optionally diagnostics) for real/complex batched tensors.

    Full matrices completes null bases; a requested rank truncates outputs.
    Values-only currently computes vectors internally and discards them.
    Input is never overwritten. Sketch generation is included in solve timing.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f'algorithm must be one of {ALGORITHMS}')
    if a.ndim < 2 or min(a.shape[-2:]) < 1:
        raise ValueError('nonempty matrix dimensions required')
    if a.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError('supported dtypes: float32/64, complex64/128')
    rank = opts.get('rank', min(a.shape[-2:]))
    if not 1 <= rank <= min(a.shape[-2:]):
        raise ValueError('rank outside [1,min(m,n)]')
    if full_matrices and rank != min(a.shape[-2:]):
        raise ValueError('full_matrices and truncated rank cannot be combined')
    for key in ('max_iter', 'max_sweeps', 'leaf_size'):
        if key in opts and opts[key] < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('oversample', 'iterations'):
        if key in opts and opts[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    shape = a.shape[:-2]
    x = a.reshape(-1, *a.shape[-2:])
    if not bool(torch.isfinite(x).all()):
        raise ValueError('input contains NaN or infinity')
    # Normalize before any sums of squares to support extreme input scales.
    scale = x.abs().amax((-2, -1), keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    stats = {'algorithm': algorithm, 'vendor_factorizations': 0, 'converged': True}
    with torch.no_grad():
        u, s, v = _raw(x / scale, algorithm, stats, opts)
        s = s * scale[:, 0, 0][:, None]
        if full_matrices:
            u = P.complete_basis(u, a.shape[-2])
            v = P.complete_basis(v, a.shape[-1])
        else:
            u, s, v = u[:, :, :rank], s[:, :rank], v[:, :, :rank]
        if not all(bool(torch.isfinite(z).all()) for z in (u, s, v)):
            raise RuntimeError('standalone solver produced nonfinite output')
        result = (None if values_only else u.reshape(*shape, *u.shape[-2:]),
                  s.reshape(*shape, s.shape[-1]),
                  None if values_only else v.mH.reshape(*shape, v.shape[-1], v.shape[-2]))
    return (*result, stats) if return_info else result


def factory(a, spec):
    """Adapter for python -m svdbench; algorithm is in spec['method']."""
    opts = {k: v for k, v in spec.items() if k not in ('name', 'method', 'mode', 'seed_offset')}
    def run():
        u, s, vh, info = svd(a, spec['method'], full_matrices=spec.get('mode') == 'full',
                             values_only=spec.get('mode') == 'values', return_info=True, **opts)
        run.diagnostics = info
        return u, s, vh
    run.metadata = {'implementation': 'standalone tensor arithmetic', 'vendor_factorizations': False,
                    'values_only_forms_vectors': True, 'rng_in_solve': True,
                    'gpu_control': 'Python loops and host convergence synchronization; no graph capture'}
    return run

# Here is CANS SVD routine from https://github.com/fallnlove/cans_svd/

def get_polynomial(Ext):
    n = len(Ext) - 1
    M = torch.zeros((n + 1, n + 1), dtype=Ext.dtype, device=Ext.device)
    for i in range(n + 1):
        for j in range(n):
            M[i, j] = Ext[i] ** (2 * j + 1)
        M[i, n] = (-1) ** (i + 1)
    return _solve_square(M, torch.ones(n + 1, dtype=Ext.dtype, device=Ext.device))


def _solve_square(m, rhs):
    """x with m x = rhs via the standalone Householder QR; RuntimeError when m is singular."""
    q, r = P.qr(m[None])
    pivots = r[0].diagonal().abs()
    if not bool(pivots.min() > P._eps(m.dtype) * len(pivots) * pivots.max()):
        raise RuntimeError('singular system')
    return P.trsm_upper(r, q.mH @ rhs[None, :, None])[0, :, 0]


def get_ext(c):
    n = len(c)
    coeffs = [(2 * j + 1) * c[j] for j in range(n)]
    rts = poly_roots(torch.tensor(coeffs[::-1], device=c.device))
    return torch.sqrt(rts)


def remez_step(Ext):
    n = len(Ext) - 1
    c = get_polynomial(Ext)
    coeffs = [c[j // 2 - 1] if j % 2 == 0 else 0 for j in range(1, 2 * n + 1)]
    p = coeffs[::-1]
    e = c[n].item()
    NewExt = torch.concatenate((Ext[0], get_ext(c[:n]), Ext[-1]))
    f = lambda x: torch.abs(sum(a * x**i for i, a in enumerate(p)) - 1) # np.abs(p(x) - 1)
    newe = torch.max(f(NewExt))
    return coeffs, NewExt, e, newe


def remez(A, B, degree):
    # Remez algorithm for finding optimal polynomial approximating the unity function f==1
    # on the segment [A, B]
    
    Ext = torch.linspace(B, A, (degree + 1) // 2 + 1)
    p = 0.0
    newe = 0
    for i in range(100):
        try:
            p, NewExt, e, newe = remez_step(Ext)
        except RuntimeError:
            return kovarik_formula(degree), 0  # if the segment converged to [A, B]=[1, 1]
        if newe < abs(e) + 1e-20:
            return p, newe
        Ext = torch.array(NewExt)
    return p, newe


def c_n_k(n, k):
    s = 1
    for i in range(n - k + 1, n + 1):
        s *= i
    for i in range(1, k + 1):
        s /= i
    return s


def kovarik_formula(degree):
    # generates polynomial of specified degree from the paper
    # Zdislav Kovarik, "SOME ITERATIVE METHODS FOR IMPROVING ORTHONORMALITY", 1970
    p = torch.zeros(degree + 1)
    p[1] += 1
    a = 1
    for i in range(1, (degree + 1) // 2):
        for j in range(2 * (i - 1) + 1, 2 * i + 1):
            a *= j / 2
        a /= i ** 2
        sign = 1
        for k in range(0, i + 1):
            p[2 * k + 1] += a * sign * c_n_k(i, k)
            sign *= -1
    return p

def poly_roots(coeffs):
    """Complex roots of coeffs[0] x^n + ... + coeffs[n] by Durand-Kerner iteration.

    Plain arithmetic instead of the eigenvalues of the companion matrix, so that no
    library eigensolver is called.
    """
    c = coeffs.to(torch.complex128)
    n = len(c) - 1
    if n < 1:
        return c[:0]
    c = c / c[0]
    radius = 1 + float(c[1:].abs().max())
    angles = 2 * math.pi * (torch.arange(n, dtype=torch.float64, device=c.device) + 0.25) / n
    z = torch.polar(torch.full((n,), radius, dtype=torch.float64, device=c.device), angles)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=c.device)
    for _ in range(500):
        value = torch.zeros_like(z)
        for coefficient in c:                       # Horner
            value = value * z + coefficient
        differences = z[:, None] - z[None, :]
        gaps = torch.where(off_diagonal, differences, torch.ones_like(differences))
        step = value / gaps.prod(1)
        z = z - step
        if bool(step.abs().max() <= 1e-14 * radius):
            break
    if torch.isnan(z).any() or torch.isinf(z).any():
        raise RuntimeError('Error in polyroot evaluation')
    return z
