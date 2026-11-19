"""Frozen V5 weights with a single, history-independent deployment path.

CVX objective is unchanged: ||A s-z||^2 + gamma^2 ||s||^2, s>=0.
An augmented NNLS solve avoids squared conditioning and OSQP warm-start history.
CPU is the canonical export backend; CUDA benchmarks use the same float64
physical projection and include host/device transfers and the CPU convex solve.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.ndimage import gaussian_filter
from .shared_runtime import SharedRuntime
from .shared_physics import SharedPhysics
from .compact_physics import sketched_rgb
from .data import angle_image


class StablePhysics(SharedPhysics):
    def __init__(self, *args):
        super().__init__(*args)
        self.q = np.ascontiguousarray(self.q, dtype=np.float64)
        self.augmented = {}

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            return cls(*(f[k] for k in ('q', 'a', 'scale', 'mask')))

    def project(self, x):
        if np.ndim(x) == 2:
            return np.stack([self.project(row) for row in x])
        return np.asarray(x, np.float64) @ self.q / self.scale

    def solve(self, z, solver='svd', gamma=0.):
        if solver != 'cvx':
            return super().solve(z, solver, gamma)
        if np.ndim(z) == 2:
            return np.stack([self.solve(row, solver, gamma) for row in z])
        if gamma not in self.augmented:
            self.augmented[gamma] = np.vstack((self.a, gamma*np.eye(self.a.shape[1])))
        # Positive rescaling of the right-hand side preserves the minimizer,
        # and keeps the stopping criterion independent of sensor brightness.
        scale = max(float(np.linalg.norm(z)), 1e-30)
        rhs = np.r_[z/scale, np.zeros(self.a.shape[1])]
        result, _ = nnls(self.augmented[gamma], rhs, maxiter=10000)
        return result*scale


class Runtime(SharedRuntime):
    def __init__(self, root, method, device='cpu'):
        super().__init__(Path(root), method)
        self.device = torch.device(device)
        if method == 'traditional':
            self.bank = [StablePhysics.load(self.root/'models/physics_final'/f'angle_{i}.npz')
                         for i in range(len(self.angles))]
            self.packed = np.ascontiguousarray(np.concatenate([m.q for m in self.bank], axis=1))
            for i, m in enumerate(self.bank):
                m.q = self.packed[:, self.offsets[i]:self.offsets[i+1]]
            self.gpu_packed = torch.from_numpy(self.packed).to(self.device) if device != 'cpu' else None
        else:
            self.net.to(self.device)
        self.get = lru_cache(maxsize=2)(lambda a: StablePhysics.load(
            self.root/'models/physics_final'/f'angle_{a}.npz'))
        if self.weight is not None:
            source=json.loads((self.root/'models/spectral_source.json').read_text(encoding='utf-8'))
            with np.load((self.root/'models'/source['path']) if not Path(source['path']).is_absolute() else Path(source['path'])) as f:
                self.weight=torch.from_numpy(f['weight'].astype(np.float64)/float(f['input_divisor'])).to(self.device)

    def project(self, x, matrix):
        x = np.asarray(x, np.float64)
        if self.device.type == 'cpu':
            return x @ matrix
        q = self.gpu_packed if self.method=='traditional' and matrix is self.packed else torch.from_numpy(matrix).to(self.device)
        return (torch.from_numpy(x).to(self.device) @ q).cpu().numpy()

    def angle(self, rgb):
        g = torch.from_numpy(angle_image(rgb,32)[None,None]).to(self.device)
        p = self.net(g).softmax(1)[0].cpu().numpy()
        return int(p.argmax()), float(p.max())

    def physical(self, x, a, confidence):
        model = self.bank[a] if self.method == 'traditional' else self.get(a)
        z = self.project(x, model.q)/model.scale
        initial = model.solve(z)
        energy = float(np.asarray(x,np.float64) @ np.asarray(x,np.float64))
        outside = float(np.sqrt(max(0., energy-float(z@z)*model.scale**2)/max(energy,1e-20)))
        action = int(self.policy['fixed_action'])
        called = self.method == 'hybrid' and outside > self.policy['noise_gate_threshold']
        if called:
            if self.selector is None:
                import lightgbm as lgb
                self.selector = lgb.Booster(model_str=(self.root/'models/selector.txt').read_text(encoding='utf-8'))
            p = self.selector.predict(model.features(x,initial,confidence)[None],num_threads=1)[0]
            if p.max() >= self.policy['selector_threshold']:
                action = int(p.argmax())
        solver,gamma = self.policy['actions'][action]
        raw = initial if solver == 'svd' else model.solve(z,solver,gamma)
        return raw, dict(solver=solver,gamma=gamma,selected_action=action,
            selector_evaluated=called,outside_residual=outside,confidence=confidence,
            residual=model.residual(np.asarray(x,np.float64),raw,z),
            closed_loop_triggered=False,closed_loop_accepted=False)

    @torch.inference_mode()
    def predict(self, rgb, exposure, profile=False):
        scale = np.float32(30000./exposure/255.)
        if self.method == 'traditional':
            x = sketched_rgb(rgb,6144,scale)
            coeff = self.project(x,self.packed)
            energy = np.add.reduceat(coeff**2*self.mask,self.offsets[:-1])
            a = int(energy.argmax()); confidence = 1.
        else:
            a,confidence = self.angle(rgb)
            if self.method == 'pure_ml':
                xx = torch.from_numpy(np.asarray(rgb,np.float64).transpose(2,0,1)[None]*float(scale)).to(self.device)
                v = torch.nn.functional.adaptive_avg_pool2d(xx,(self.side,self.side)).flatten(1)
                raw = torch.nn.functional.linear(v,self.weight)[0].cpu().numpy()
                return float(self.angles[a]),raw,dict(solver='linear_readout',gamma=0.,confidence=confidence,
                    selector_evaluated=False,closed_loop_triggered=False,closed_loop_accepted=False,residual=None)
            x = sketched_rgb(rgb,6144,scale)
        raw,info = self.physical(x,a,confidence)
        if self.method == 'hybrid' and info['residual'] > .35 and confidence < .55:
            import bm3d
            den = np.stack([bm3d.bm3d(rgb[:,:,c].astype(float)/255.,sigma_psd=.02) for c in range(3)],axis=-1)
            corrected = np.uint8(np.clip((den+.35*(den-gaussian_filter(den,sigma=(1,1,0))))*255.,0,255))
            aa,cc = self.angle(corrected)
            ss,ii = self.physical(sketched_rgb(corrected,6144,scale),aa,cc)
            residual = self.get(aa).residual(np.asarray(x,np.float64),ss)
            info['closed_loop_triggered'] = True
            if residual < info['residual']:
                a,raw,info = aa,ss,dict(ii,residual=residual,closed_loop_triggered=True,closed_loop_accepted=True)
        return float(self.angles[a]),raw,info

    def predict_many(self, images, exposures):
        return [self.predict(rgb,exposure) for rgb,exposure in zip(images,exposures)]
