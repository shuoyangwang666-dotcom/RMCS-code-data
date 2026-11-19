import numpy as np

def normalized_spectrum(x):
    x = np.maximum(np.asarray(x), 0.)
    return x / np.maximum(np.max(x, axis=-1, keepdims=True), 1e-12)
