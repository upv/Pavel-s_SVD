"""Charts, tables and a static PDF report from bench_matrix_types.py output.

    python3 benchmarks/matrix_types/make_report.py benchmarks/matrix_types/results/<run>

Writes *.png charts, report.html and report.pdf into the same directory. The PDF is printed
with headless Chromium, found via $CHROME, then chromium / chromium-browser / google-chrome.
"""
import html, json, math, os, shutil, subprocess, sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FAMILY = {
    'torch.linalg.svd': 'Reference (cuSOLVER)',
    'bidiag_qr': 'Golub-Kahan bidiagonal', 'bidiag_dc': 'Golub-Kahan bidiagonal', 'recipes.golub_kahan': 'Golub-Kahan bidiagonal',
    'jacobi': 'One-sided Jacobi', 'block_jacobi': 'One-sided Jacobi',
    'qr': 'QR then Jacobi', 'tsqr': 'QR then Jacobi', 'recipes.qr_then_svd': 'QR then Jacobi',
    'gram_jacobi': 'Gram eigensolver', 'gram_dc': 'Gram eigensolver',
    'polar_qdwh': 'Polar decomposition', 'polar_newton_schulz': 'Polar decomposition', 'polar_halley': 'Polar decomposition',
    'polar_polynomial': 'Polar decomposition', 'polar_cans': 'Polar decomposition',
    'randomized': 'Randomized / Krylov', 'subspace': 'Randomized / Krylov', 'block_krylov': 'Randomized / Krylov',
    'recipes.randomized': 'Randomized / Krylov',
    'lanczos': 'Lanczos bidiagonal', 'recipes.lanczos': 'Lanczos bidiagonal',
}
FAM = list(dict.fromkeys(FAMILY.values()))
COLORS = ['#222222', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MARKERS = ['o', 's', '^', 'D', '*', 'v', 'P']
SIZES = [32, 64, 128, 256]

d = sys.argv[1]
raw = json.load(open(f'{d}/results.json'))
env = json.load(open(f'{d}/env.json'))
label = lambda impl: impl.split(':', 1)[1] if impl.startswith('blocks.svd:') else impl
for r in raw:
    r['label'] = label(r['impl'])
    r['family'] = FAMILY[r['label']]
    r['prec'] = '64' if r['dtype'].endswith('64') and r['dtype'] != 'complex64' or r['dtype'] == 'complex128' else '32'
    r['orth'] = max(r['orth_u'], r['orth_v']) if r.get('orth_u') is not None else None
    r['gflop'] = r['total_flops'] / 1e9 if r.get('total_flops') else None
ok = lambda r: r['status'].startswith('ok')
TYPES = list(env['types'])
METHODS = sorted({r['label'] for r in raw}, key=lambda m: (FAM.index(FAMILY[m]), m))
R = {(r['type'], r['label'], r['n'], r['prec']): r for r in raw}


def at(kind, m, prec='32', n=None):
    """Record at the requested n, else at the largest n that ran (n=256 is skipped for slow methods)."""
    for k in ([n] if n else SIZES[::-1]):
        r = R.get((kind, m, k, prec))
        if r and ok(r):
            return r
    return None


NMAX = max(r['n'] for r in raw)
style = {}
for f in FAM:
    for i, m in enumerate([m for m in METHODS if FAMILY[m] == f]):
        style[m] = dict(color=COLORS[FAM.index(f)], marker=MARKERS[i % len(MARKERS)])


# --- charts ------------------------------------------------------------------------------
def heatmap(ax, field, title, fmt, log=True, vmin=None, vmax=None, cmap='viridis_r', n=NMAX):
    M = np.full((len(METHODS), len(TYPES)), np.nan)
    smaller = np.zeros_like(M, dtype=bool)
    for i, m in enumerate(METHODS):
        for j, t in enumerate(TYPES):
            r = R.get((t, m, n, '32'))
            if not (r and ok(r)):
                r = at(t, m)
                smaller[i, j] = r is not None
            if r and r.get(field) is not None:
                M[i, j] = r[field]
    V = np.log10(np.maximum(M, 1e-17)) if log else M
    im = ax.imshow(V, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not math.isnan(M[i, j]):
                v = V[i, j]
                lo, hi = im.get_clim()
                dark = (v - lo) / (hi - lo + 1e-12) > 0.55
                ax.text(j, i, fmt(M[i, j]) + ('*' if smaller[i, j] else ''), ha='center', va='center', fontsize=6.5,
                        color='white' if dark else 'black')
            else:
                ax.text(j, i, 'fail', ha='center', va='center', fontsize=6.5, color='#c00')
    ax.set_xticks(range(len(TYPES)), TYPES, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(METHODS)), METHODS, fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


sci = lambda v: f'{v:.0e}'.replace('e-0', 'e-').replace('e+0', 'e')
CHARTS = [
    ('recon_heatmap', 'recon_err', f'Reconstruction error ||A - USV*||/||A||, float32, n={NMAX}', sci, True, -8, 0),
    ('orth_heatmap', 'orth', f'Orthogonality error max(||U*U - I||, ||V*V - I||), float32, n={NMAX}', sci, True, -7, 1),
    ('sverr_heatmap', 'sv_err_rel', f'Max relative error of resolvable singular values, float32, n={NMAX}', sci, True, -8, 0),
    ('time_heatmap', 'median_ms', f'Median GPU time, ms, float32, n={NMAX}', lambda v: f'{v:.3g}', True, None, None),
    ('flops_heatmap', 'gflop', f'Executed GFLOP, float32, n={NMAX}', lambda v: f'{v:.2g}', True, None, None),
]
for name, field, title, fmt, log, lo, hi in CHARTS:
    fig, ax = plt.subplots(figsize=(10, 7.2))
    im = heatmap(ax, field, title, fmt, log, lo, hi, cmap='viridis' if field in ('median_ms', 'gflop') else 'viridis_r')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('log10')
    fig.text(0.01, 0.005, f'* value at the largest n that ran (n={NMAX} skipped after a slow n=128)', fontsize=7)
    fig.tight_layout()
    fig.savefig(f'{d}/{name}.png', dpi=130)
    plt.close(fig)


def grid(field, name, title, ylabel):
    cols = 3
    rows = math.ceil(len(TYPES) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.3 * rows), sharex=True)
    for ax, t in zip(axes.flat, TYPES):
        for m in METHODS:
            pts = [(n, R[(t, m, n, '32')][field]) for n in SIZES
                   if (t, m, n, '32') in R and ok(R[(t, m, n, '32')]) and R[(t, m, n, '32')].get(field)]
            if pts:
                ax.plot(*zip(*pts), lw=1.2, ms=4, label=m, **style[m])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(SIZES, [str(s) for s in SIZES])
        ax.set_title(t, fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(TYPES):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=6, fontsize=7.5, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(f'{d}/{name}.png', dpi=110)
    plt.close(fig)


grid('recon_err', 'recon_vs_n', 'Reconstruction error vs n per matrix type (float32)', 'rel. error')
grid('sv_err_rel', 'sverr_vs_n', 'Max relative error of resolvable singular values vs n (float32)', 'rel. error')
grid('median_ms', 'time_vs_n', 'Median GPU time vs n per matrix type (float32)', 'ms')

# --- tables + HTML --------------------------------------------------------------------------
esc = html.escape
num = lambda v, f: '' if v is None else f(v)
ms = lambda v: f'{v:,.0f}' if v >= 100 else f'{v:.1f}' if v >= 10 else f'{v:.2f}'
e1 = lambda v: f'{v:.1e}'


def iters(r):
    st, it, out = r.get('stats') or {}, r.get('iters') or {}, []
    for k, nm in (('polar_iterations', 'polar it'), ('block_sweeps', 'outer sweeps'), ('lanczos_steps', 'Lanczos steps'),
                  ('power_iterations', 'power it')):
        if k in st:
            out.append(f'{nm} {st[k]}')
    for k, nm in (('bdsqr_iterations', 'bdsqr it'), ('jacobi_svd_sweeps', 'Jacobi sweeps'), ('jacobi_eigh_sweeps', 'eigh sweeps')):
        v = it.get(k)
        if v is not None:
            out.append(f"{nm} {v['total']}/{v['calls']} calls" if isinstance(v, dict) else f'{nm} {v}')
    if st.get('converged') is False:
        out.append('NOT converged')
    return ', '.join(out)


def type_table(t):
    rows = []
    for m in METHODS:
        for prec in ('32', '64'):
            r = at(t, m, prec)
            if r is None:
                bad = [x for x in raw if x['type'] == t and x['label'] == m and x['prec'] == prec and x['status'] == 'failed']
                if bad:
                    rows.append(f'<tr class="fail"><td>{esc(m)}</td><td>{bad[0]["dtype"]}</td>'
                                f'<td colspan="8">failed at n={bad[0]["n"]}: {esc(bad[0]["error"][:120])}</td></tr>')
                continue
            star = '' if r['n'] == NMAX else f' (n={r["n"]})'
            est = ' est.' if r.get('flops_estimated') else ''
            rows.append(f'<tr><td>{esc(m)}{star}</td><td>{r["dtype"]}</td><td>{ms(r["median_ms"])}</td>'
                        f'<td>{num(r.get("gflop"), lambda v: f"{v:.2f}")}{est}</td><td>{e1(r["recon_err"])}</td>'
                        f'<td>{e1(r["orth"])}</td><td>{e1(r["sv_err_rel"])}</td><td>{e1(r["sv_err_norm"])}</td>'
                        f'<td>{r["n_resolvable"]}</td><td class="it">{esc(iters(r))}</td></tr>')
    return ('<table><tr><th>method</th><th>dtype</th><th>time, ms</th><th>GFLOP</th><th>recon.</th><th>orth.</th>'
            '<th>σ rel. err</th><th>σ err / σ₁</th><th>resolv. σ</th><th>iterations</th></tr>' + ''.join(rows) + '</table>')


findings = open(f'{d}/findings.html').read() if os.path.exists(f'{d}/findings.html') else ''
img = lambda n, cap='': f'<figure><img src="{n}.png"><figcaption>{cap}</figcaption></figure>'
types_html = ''.join(f'<tr><td><b>{t}</b></td><td>{esc(env["types"][t])}</td></tr>' for t in TYPES)
body = f"""
<h1>SVD accuracy, GPU time and FLOPs across matrix types</h1>
<p class="meta">{esc(env['gpu'])} · PyTorch {esc(env['torch'])} · CUDA {esc(str(env['cuda']))} · n = {', '.join(map(str, env['sizes']))}
· {env['warmup']} warm-up + {env['reps']} timed runs (median) · square n×n matrices, one seeded matrix per (type, n)</p>
{findings}
<h2>Matrix types</h2><table class="types">{types_html}</table>
<h2>Metrics</h2>
<ul>
<li><b>recon.</b> ‖A − U diag(s) V*‖_F / ‖A‖_F, computed in float64 from the returned factors.</li>
<li><b>orth.</b> max(‖U*U − I‖_F, ‖V*V − I‖_F).</li>
<li><b>σ rel. err</b> max |sᵢ − σᵢ| / σᵢ over <i>resolvable</i> singular values σᵢ &gt; n·ε·σ₁ (ε of the working precision);
σ are float64 singular values of the matrix after the cast. <b>σ err / σ₁</b> is the normwise error max |sᵢ − σᵢ| / σ₁, over all values.</li>
<li><b>GFLOP</b> executed FLOPs: matmul FLOPs measured with FlopCounterMode plus exact element-wise counts of the repository
primitives with actual iteration counts (method of benchmarks/gpu_flops). torch.linalg.svd: textbook estimate 21n³.
Complex matrices: counts ×4 as real-arithmetic equivalents (approximate).</li>
<li><b>time</b> CUDA-event median, includes Python and launch overhead.</li>
</ul>
<h2>Accuracy at n = {NMAX}</h2>
{img('recon_heatmap')}{img('orth_heatmap')}{img('sverr_heatmap')}
<h2>Cost at n = {NMAX}</h2>
{img('time_heatmap')}{img('flops_heatmap')}
<h2>Scaling with n</h2>
{img('recon_vs_n')}{img('sverr_vs_n')}{img('time_vs_n')}
<h2>Tables per matrix type (n = {NMAX}; a smaller n is marked where n = {NMAX} was skipped)</h2>
""" + ''.join(f'<h3>{t}</h3><p class="desc">{esc(env["types"][t])}</p>{type_table(t)}' for t in TYPES)

css = """body{font:11px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1a1a1a;margin:24px;max-width:1100px}
h1{font-size:20px}h2{font-size:15px;margin-top:22px;border-bottom:1px solid #ddd;padding-bottom:3px}h3{font-size:13px;margin:16px 0 2px}
.meta,.desc{color:#555}table{border-collapse:collapse;width:100%;margin:6px 0}td,th{border-bottom:1px solid #e3e3e3;padding:2px 5px;text-align:right}
td:first-child,th:first-child,td.it{text-align:left}td:nth-child(2){text-align:left;color:#666}.it{font-size:9.5px;color:#555}
table.types td{text-align:left}tr.fail td{color:#b00;text-align:left}figure{margin:8px 0;page-break-inside:avoid}
img{max-width:100%}figcaption{color:#666}table,tr{page-break-inside:avoid}.find li{margin:3px 0}"""
open(f'{d}/report.html', 'w').write(f'<!doctype html><html><head><meta charset="utf-8"><title>SVD across matrix types</title>'
                                    f'<style>{css}</style></head><body>{body}</body></html>')

chrome = os.environ.get('CHROME') or next((shutil.which(c) for c in ('chromium', 'chromium-browser', 'google-chrome')
                                           if shutil.which(c)), None)
if chrome:
    subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={os.path.abspath(d)}/report.pdf', f'file://{os.path.abspath(d)}/report.html'],
                   check=True, capture_output=True)
    print('wrote', f'{d}/report.pdf')
else:
    print('no Chromium found: wrote report.html only')
