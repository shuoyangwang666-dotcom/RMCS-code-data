import numpy as np
import scipy.linalg as la

def svd_inverse(matrix, gamma=0.):
    u, s, vt = la.svd(matrix, full_matrices=False, check_finite=True)
    if gamma < 0:
        raise ValueError('gamma must be nonnegative')
    if gamma == 0:
        cutoff = max(matrix.shape) * np.finfo(float).eps * s[0]
        factors = np.divide(1., s, out=np.zeros_like(s), where=s > cutoff)
    else:
        factors = s / (s * s + gamma * gamma)
    return (vt.T * factors) @ u.T
