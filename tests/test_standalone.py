"""Independent references and adversarial contracts for every standalone family."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import scipy.linalg
import torch
from svdbench.standalone import svd, ALGORITHMS
from svdbench.standalone import primitives as P
from svdbench.standalone.iterative import DenseOperator, lanczos_svd, rank_one_update, append_columns
from svdbench.standalone.tensors import tucker, tucker_reconstruct, tensor_train, tensor_train_reconstruct
from svdbench.matrices import make_matrix

DEVICE = os.environ.get('SVD_TEST_DEVICE', 'cpu')
torch.set_num_threads(1)


def tensor(a):
    return torch.as_tensor(a, device=DEVICE)


def error(a, b):
    scale = a.abs().max()
    if bool(scale == 0):
        return float(b.abs().max())
    aa, bb = a / scale, b / scale
    return float((aa - bb).abs().square().sum().sqrt() / aa.abs().square().sum().sqrt())


class Standalone(unittest.TestCase):
    def check(self, a, name, tol, **kw):
        u, s, vh = svd(a, name, **kw)
        r = len(s) if s.ndim == 1 else s.shape[-1]
        recon = (u[..., :r] * s[..., None, :]) @ vh[..., :r, :]
        self.assertLess(error(a, recon), tol, name + ' reconstruction')
        for q in [u, vh.mH]:
            eye = torch.eye(q.shape[-1], device=a.device, dtype=a.dtype)
            self.assertLess(float((q.mH @ q - eye).abs().max()), tol * 5, name + ' orthogonality')
        aa = a.cpu().numpy().reshape(-1, *a.shape[-2:])
        expected = np.stack([scipy.linalg.svdvals(x.astype(np.complex128 if np.iscomplexobj(x) else np.float64)) for x in aa])
        actual = s.cpu().numpy().reshape(expected.shape)
        self.assertLess(float(np.max(abs(actual - expected)) / max(np.max(expected), np.finfo(float).tiny)), tol, name + ' spectrum')
        return u, s, vh

    def test_all_families_real_complex_tall_wide_batched(self):
        for dt in ('float32', 'float64', 'complex64', 'complex128'):
            tol = 3e-4 if dt.endswith('32') or dt == 'complex64' else 2e-10
            for m, n in ((13, 7), (7, 13)):
                a = tensor(make_matrix(dict(m=m, n=n, batch=2, dtype=dt, family='gaussian')))
                for name in ALGORITHMS:
                    with self.subTest(dtype=dt, shape=(m, n), algorithm=name):
                        self.check(a, name, tol, tile_rows=7)

    def test_zero_identity_rank_deficient_clustered_and_scaled(self):
        matrices = [np.zeros((9, 5)), np.eye(7), np.diag([0., 2., 0., 3., 1.])]
        matrices += [make_matrix(dict(m=15, n=7, dtype='float64', family='low_rank', rank=3, tail=0)),
                     make_matrix(dict(m=15, n=7, dtype='float64', family='clustered'))]
        base = make_matrix(dict(m=11, n=5, dtype='float64', family='gaussian'))
        matrices += [base * 1e-150, base * 1e150]
        for i, data in enumerate(matrices):
            for name in ALGORITHMS:
                with self.subTest(matrix=i, algorithm=name):
                    self.check(tensor(data), name, 2e-9)

    def test_singleton_and_full_null_bases_values_only(self):
        for shape in ((1, 1), (8, 1), (1, 8), (10, 4), (4, 10)):
            a = tensor(make_matrix(dict(m=shape[0], n=shape[1], family='gaussian', dtype='complex128')))
            for name in ALGORITHMS:
                with self.subTest(shape=shape, algorithm=name):
                    u, s, vh = self.check(a, name, 2e-10, full_matrices=True)
                    self.assertEqual(u.shape, (shape[0], shape[0]))
                    self.assertEqual(vh.shape, (shape[1], shape[1]))
                    none, ss, none2 = svd(a, name, values_only=True)
                    self.assertIsNone(none); self.assertIsNone(none2)
                    self.assertLess(float((s - ss).abs().max()), 1e-12)

    def test_ill_conditioned_accuracy_envelopes(self):
        a = tensor(make_matrix(dict(m=24, n=9, dtype='float64', family='geometric', condition=1e8)))
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                # Gram deliberately drops values below its eigenvalue error floor.
                if name.startswith('gram'):
                    u, s, vh, info = svd(a, name, return_info=True)
                    self.assertLess(error(a, (u * s) @ vh), 2e-7)
                    self.assertGreater(info['gram_discarded_values'], 0)
                    self.assertLess(float((u.mH @ u - torch.eye(9, device=DEVICE)).abs().max()), 0.1)
                else:
                    self.check(a, name, 2e-8)

    def test_low_rank_and_warm_start(self):
        a = tensor(make_matrix(dict(m=40, n=25, dtype='complex128', family='low_rank', rank=4, tail=0)))
        for name in ('randomized', 'subspace', 'block_krylov', 'lanczos'):
            u, s, vh = svd(a, name, rank=4, oversample=3, iterations=2)
            self.assertLess(error(a, (u * s) @ vh), 1e-10, name)
            self.assertEqual(s.shape, (4,))
        _, _, initial = svd(a, 'jacobi', rank=7)
        u, s, vh, info = svd(a, 'subspace', rank=4, oversample=3, initial_basis=initial.mH,
                              iterations=1, return_info=True)
        self.assertTrue(info['warm_started'])
        self.assertLess(error(a, (u * s) @ vh), 1e-10)

    def test_matrix_free_sparse_lanczos_and_breakdown(self):
        a = torch.diag(tensor(np.array([5., 4., 3., 2., 1., 0.])))
        class Sparse:
            shape, dtype, device = a.shape, a.dtype, a.device
            mat = a.to_sparse()
            def matmat(self, x): return torch.sparse.mm(self.mat, x)
            def rmatmat(self, x): return torch.sparse.mm(self.mat.transpose(0, 1), x)
        u, s, v, info = lanczos_svd(Sparse(), rank=6, steps=6)
        self.assertLess(error(a, (u * s) @ v.mH), 1e-11)
        self.assertEqual(info['matmat_calls'], 7)

    def test_updates_and_column_appends(self):
        for dtype in (torch.float64, torch.complex128):
            gen = torch.Generator(device=DEVICE).manual_seed(32)
            a = torch.randn(12, 7, dtype=dtype, device=DEVICE, generator=gen)
            u, s, vh = svd(a, 'jacobi')
            left = torch.randn(12, dtype=dtype, device=DEVICE, generator=gen)
            right = torch.randn(7, dtype=dtype, device=DEVICE, generator=gen)
            un, sn, vn = rank_one_update(u, s, vh, left, right)
            self.assertLess(error(a + left[:, None] * right.conj(), (un * sn) @ vn), 1e-11)
            columns = torch.randn(12, 3, dtype=dtype, device=DEVICE, generator=gen)
            un, sn, vn = append_columns(u, s, vh, columns)
            self.assertLess(error(torch.cat([a, columns], 1), (un * sn) @ vn), 1e-11)
            un, sn, vn = rank_one_update(u, s, vh, left * 0, right)
            self.assertLess(error(a, (un * sn) @ vn), 1e-11)

    def test_tucker_sequential_and_tensor_train(self):
        x = tensor(np.random.default_rng(7).normal(size=(5, 4, 3)))
        for seq in (False, True):
            core, factors, info = tucker(x, list(x.shape), sequential=seq)
            self.assertLess(error(x, tucker_reconstruct(core, factors)), 1e-11)
            self.assertEqual(len(info), 3)
        cores, _ = tensor_train(x, [5, 3])
        self.assertLess(error(x, tensor_train_reconstruct(cores)), 1e-11)
        # Known rank-one tensor: all three compression paths should be exact.
        low = x[:, :1, :1] * x[:1, :, :1] * x[:1, :1, :]
        for seq in (False, True):
            core, factors, _ = tucker(low, [1, 1, 1], sequential=seq)
            self.assertLess(error(low, tucker_reconstruct(core, factors)), 1e-11)
        cores, _ = tensor_train(low, [1, 1])
        self.assertLess(error(low, tensor_train_reconstruct(cores)), 1e-11)

    def test_cans_helpers_without_vendor_calls(self):
        from svdbench.standalone.solvers import poly_roots, get_polynomial, _solve_square
        roots = poly_roots(torch.tensor([1., -6., 11., -6.], dtype=torch.float64))
        np.testing.assert_allclose(np.sort(roots.real.numpy()), [1, 2, 3], atol=1e-10)
        self.assertLess(float(roots.imag.abs().max()), 1e-10)
        m = torch.tensor([[4., 1., 0.], [1., 3., 1.], [0., 1., 2.]], dtype=torch.float64)
        rhs = torch.tensor([1., 2., 3.], dtype=torch.float64)
        np.testing.assert_allclose((m @ _solve_square(m, rhs)).numpy(), rhs.numpy(), atol=1e-12)
        c = get_polynomial(torch.linspace(1.5, .5, 3, dtype=torch.float64))
        self.assertEqual(tuple(c.shape), (3,))
        self.assertTrue(bool(torch.isfinite(c).all()))

    def test_no_vendor_factorizations_and_immutable_input(self):
        forbidden = ['svd', 'svdvals', 'qr', 'eigh', 'eig', 'eigvalsh', 'cholesky', 'solve', 'inv']
        from contextlib import ExitStack
        a = tensor(np.random.default_rng(9).normal(size=(9, 6)))
        saved = a.clone()
        with ExitStack() as stack:
            for name in forbidden:
                stack.enter_context(patch.object(torch.linalg, name, side_effect=AssertionError('vendor factorization called')))
            for name in ALGORITHMS:
                svd(a, name)
        self.assertTrue(torch.equal(a, saved))
        for path in Path('svdbench/standalone').glob('*.py'):
            if path.name.startswith('.'):
                continue
            source = path.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(n.name not in ('numpy', 'scipy', 'ctypes') for n in node.names), str(path))
                if isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, 'linalg', str(path))

    def test_secular_deflation_and_recursive_merges(self):
        from svdbench.standalone.divide_conquer import rank_one_eigh, tridiagonal_dc
        for n in (1, 8, 32):
            d = torch.arange(n, device=DEVICE, dtype=torch.float64) // 3
            for zero in (False, True):
                z = torch.zeros_like(d) if zero else torch.arange(1,n+1,device=DEVICE,dtype=d.dtype) / n
                h = torch.diag(d) + z[:,None] * z
                stats = {}
                w,v = rank_one_eigh(d,z,stats)
                self.assertLess(error(h,(v*w)@v.T), 1e-12)
                self.assertLess(float((v.T@v-torch.eye(n,device=DEVICE)).abs().max()),1e-12)
        d = tensor(np.random.default_rng(19).normal(size=32))
        e = tensor(np.random.default_rng(20).normal(size=31))
        h = torch.diag(d)+torch.diag(e,1)+torch.diag(e,-1)
        stats={}
        w,v=tridiagonal_dc(d,e,stats)
        self.assertLess(error(h,(v*w)@v.T),1e-11)
        self.assertGreater(stats['secular_merges'],1)

    def test_multidimensional_batches_and_complex_tensor(self):
        a=tensor(np.random.default_rng(22).normal(size=(2,3,9,5)))
        for name in ALGORITHMS:
            u,s,vh=svd(a,name)
            self.assertEqual(s.shape,(2,3,5))
            self.assertLess(error(a,(u*s[...,None,:])@vh),1e-10)
        x=tensor(np.random.default_rng(21).normal(size=(4,3,2))+1j*np.random.default_rng(22).normal(size=(4,3,2)))
        core,factors,_=tucker(x,[4,3,2],sequential=True)
        self.assertLess(error(x,tucker_reconstruct(core,factors)),1e-11)
        cores,_=tensor_train(x,[4,2])
        self.assertLess(error(x,tensor_train_reconstruct(cores)),1e-11)

    def test_invalid_input_and_nonconvergence(self):
        a = tensor(np.random.default_rng(0).normal(size=(12, 9)))
        with self.assertRaises(RuntimeError): svd(a, 'jacobi', max_sweeps=1)
        with self.assertRaises(RuntimeError): svd(a, 'bidiag_qr', max_iter=1)
        with self.assertRaises(RuntimeError): svd(a, 'polar_newton_schulz', max_iter=1)
        for kw in ({'rank': 0}, {'iterations': -1}, {'oversample': -1}, {'leaf_size': 0}):
            with self.assertRaises(ValueError): svd(a, **kw)
        with self.assertRaises(ValueError): svd(a * float('nan'))
        with self.assertRaises(TypeError): svd(a.to(torch.int32))

if __name__ == '__main__':
    unittest.main()
