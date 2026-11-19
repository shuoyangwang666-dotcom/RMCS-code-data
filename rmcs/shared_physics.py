"""One sensor basis serves angle matching and spectral inversion; native QP cache."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from .compact_physics import CompactPhysics

class SharedPhysics(CompactPhysics):
    def __init__(self, q, a, scale, mask):
        super().__init__(q,a,scale,np.empty((0,0),np.float32),np.empty((0,0),np.float32))
        self.mask=np.asarray(mask,bool); self.qps={}

    @classmethod
    def from_compact(cls, model, rcond=1e-4):
        u,s,vt=la.svd(model.a,full_matrices=False)
        return cls((model.q.astype(np.float64)@u).astype(np.float32),s[:,None]*vt,
                   model.scale,s>s[0]*rcond)

    @classmethod
    def fit(cls,x,spectra,rcond=1e-4):
        return cls.from_compact(CompactPhysics.fit(x,spectra,rcond),rcond)

    def solve(self,z,solver='svd',gamma=0.):
        if solver!='cvx':return super().solve(z,solver,gamma)
        import osqp
        if np.ndim(z)!=1:return np.stack([self.solve(row,solver,gamma) for row in z])
        gamma=float(gamma)
        if gamma not in self.qps:
            n=self.a.shape[1]; gram=self.a.T@self.a
            qp=osqp.OSQP()
            qp.setup(P=sparse.csc_matrix(2*((gram+gram.T)/2+gamma**2*np.eye(n))),
                q=-2*self.a.T@z,A=sparse.eye(n,format='csc'),l=np.zeros(n),u=np.full(n,np.inf),
                eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True,verbose=False)
            self.qps[gamma]=qp
        qp=self.qps[gamma];qp.update(q=-2*self.a.T@z)
        result=qp.solve()
        if result.info.status_val not in (1,2) or result.x is None:
            raise RuntimeError(f'OSQP: {result.info.status}')
        return result.x.copy()

    def save(self,path):
        np.savez(path,q=self.q,a=self.a,scale=self.scale,mask=self.mask)

    @classmethod
    def load(cls,path):
        with np.load(path) as f:return cls(**{k:f[k] for k in ('q','a','scale','mask')})
