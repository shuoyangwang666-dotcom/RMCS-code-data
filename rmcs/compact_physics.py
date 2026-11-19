import numpy as np
import scipy.linalg as la
from functools import lru_cache
from .physics import svd_inverse
from .common import normalized_spectrum

def pooled_rgb(rgb, side, scale=1.):
    rgb = np.asarray(rgb, dtype=np.float32)
    h, w, c = rgb.shape
    if c != 3 or h % side or w % side:
        raise ValueError('Expected divisible square RGB ROI.')
    return (rgb.reshape(side, h//side, side, w//side, 3).mean((1, 3))
            .reshape(-1) * np.float32(scale / np.sqrt(3*side*side)))

def sketch_map(input_dim,dimension,seed=13):
    rng=np.random.default_rng(seed)
    buckets=rng.integers(0,24576,size=input_dim,dtype=np.int64)%dimension
    signs=(2*rng.integers(0,2,size=input_dim)-1).astype(np.float32)
    return buckets,signs

def sketched_rgb(rgb,dimension,scale=1.):
    values=np.asarray(rgb,dtype=np.float32).reshape(-1)
    buckets,signs=sketch_map(len(values),dimension)
    return (np.bincount(buckets,weights=values*signs,minlength=dimension)
            * (float(scale)/np.sqrt(len(values)))).astype(np.float32)

class CompactPhysics:
    def __init__(self, q, a, scale, effective, initial):
        self.q, self.a, self.scale = q, a, float(scale)
        self.effective, self.initial = effective, initial
        self.operators = {('svd', 0.): svd_inverse(a)}
        self._cvx = None

    @classmethod
    def fit(cls, x, spectra, rcond=1e-4):
        # Float64 QR avoids amplifying float32 orthogonality error in inversion.
        q, r = la.qr(np.asarray(x.T, dtype=np.float64), mode='economic')
        op = r @ np.linalg.pinv(spectra.T.astype(np.float64), rcond=rcond)
        scale = float(la.svdvals(op)[0])
        a = op / scale
        u, singular, _ = la.svd(a, full_matrices=False)
        effective = (q @ u[:, singular > singular[0]*rcond]).astype(np.float32)
        # Folding is offline only. Initial spectrum remains the same SVD map.
        initial = (svd_inverse(a) @ q.T / scale).astype(np.float32)
        return cls(q.astype(np.float32), a, scale, effective, initial)

    def project(self, x):
        return (x @ self.q).astype(np.float64) / self.scale

    def solve(self, z, solver='svd', gamma=0.):
        if solver in ('svd', 'tikhonov'):
            key = (solver, float(gamma))
            if key not in self.operators:
                self.operators[key] = svd_inverse(self.a, gamma)
            return z @ self.operators[key].T
        if solver != 'cvx':
            raise ValueError(solver)
        import cvxpy as cp
        if self._cvx is None:
            n = self.a.shape[1]
            s = cp.Variable(n, nonneg=True)
            b = cp.Parameter(n)
            g = cp.Parameter(nonneg=True)
            gram = self.a.T @ self.a
            problem = cp.Problem(cp.Minimize(cp.quad_form(s, cp.psd_wrap((gram+gram.T)/2))
                                             -2*b@s + g*cp.sum_squares(s)))
            self._cvx = s, b, g, problem
        s, b, g, problem = self._cvx
        b.value = self.a.T @ z
        g.value = gamma**2
        problem.solve(solver='OSQP', warm_start=True, eps_abs=1e-6, eps_rel=1e-6,
                      max_iter=20000, polishing=False, verbose=False)
        if s.value is None or problem.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(str(problem.status))
        return np.asarray(s.value).copy()

    def residual(self, x, s, z=None):
        z = self.project(x) if z is None else z
        outside = max(0., float(x@x) - float(z@z)*self.scale**2)
        return float(np.sqrt(outside + self.scale**2*np.sum((self.a@s-z)**2)) /
                     max(float(la.norm(x)), 1e-12))

    def features(self, x, initial=None, confidence=1.):
        raw = self.initial @ x if initial is None else initial
        signed = raw / max(float(np.max(np.abs(raw))), 1e-12)
        shape = normalized_spectrum(raw)
        return np.r_[signed, np.mean(raw < 0), np.mean(np.abs(np.diff(shape))),
                     np.mean(np.diff(shape, n=2)**2), np.log10(max(float(la.norm(x)), 1e-12)),
                     np.log10(max(float(np.max(np.abs(raw))), 1e-12)), confidence].astype(np.float32)

    def save(self, path):
        np.savez(path, q=self.q, a=self.a, scale=self.scale,
                 effective=self.effective, initial=self.initial)

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(**{k: f[k] for k in ('q', 'a', 'scale', 'effective', 'initial')})

def match_angles(x, bank):
    norm = np.sum(x*x, axis=-1)
    scores = np.stack([np.maximum(0., norm-np.sum((x @ m.effective)**2, axis=-1))
                       / np.maximum(norm, 1e-20) for m in bank], axis=-1)
    return scores.argmin(-1), scores
