import numpy as np
from scipy import ndimage as ndi
def initialize(f, width=2.3):
    signed = ndi.distance_transform_edt(f > 0) - ndi.distance_transform_edt(f < 0)
    s = np.tanh(signed / width).astype(np.float32)
    g = gradient(s)
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    v = np.sqrt(np.maximum(0.0, 1.0 - s * s))[..., None] * g / np.maximum(norm, 1e-07)
    n = np.concatenate([v, s[..., None]], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    return n.astype(np.float32)

def gradient(s):
    return np.stack([(np.roll(s, -1, axis=k) - np.roll(s, 1, axis=k)) * 0.5 for k in range(s.ndim)], axis=-1)

def old_step(n, f, J=1.0, h=2.0, D=1.0, alpha=1.0):
    dim = f.ndim
    H = sum((np.roll(n, 1, axis=k) + np.roll(n, -1, axis=k) for k in range(dim))) * J
    H[..., :dim] += D * gradient(n[..., dim])
    H[..., dim] += h * f
    hn = np.linalg.norm(H, axis=-1, keepdims=True)
    proposal = np.divide(H, np.maximum(hn, 1e-12), out=np.zeros_like(H), where=hn > 1e-12)
    proposal = np.where(hn > 1e-12, proposal, n)
    if alpha != 1.0:
        proposal = (1.0 - alpha) * n + alpha * proposal
        proposal /= np.maximum(np.linalg.norm(proposal, axis=-1, keepdims=True), 1e-12)
    return proposal.astype(np.float32)

def derivative(n, axis, p):
    d = n.ndim - 1
    b, t, u = p
    a = (0.5 - 2 * b) * (np.roll(n, -1, axis) - np.roll(n, 1, axis)) + b * (np.roll(n, -2, axis) - np.roll(n, 2, axis))
    for j in range(d):
        if j != axis:
            a = (1 - 2 * t - 2 * u) * a + t * (np.roll(a, 1, j) + np.roll(a, -1, j)) + u * (np.roll(a, 2, j) + np.roll(a, -2, j))
    return a

def tensor(n, p):
    g = [derivative(n, k, p) for k in range(n.ndim - 1)]
    return np.array([[np.sum(a * b, dtype=np.float64) for b in g] for a in g])
