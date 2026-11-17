import numpy as np
from scipy.special import ellipe
def geometry(row):
    dim = int(row['dim'])
    size = int(row['size'])
    R = float(row['R'])
    kind = row['family']
    rep = int(row['rep'])
    seed = int(row['seed'])
    spacing = float(row.get('spacing', 1))
    rng = np.random.default_rng(seed)
    x = np.indices((size,) * dim, dtype=np.float32) - (size - 1) / 2
    shift = rng.uniform(-0.45, 0.45, dim)
    for k in range(dim):
        x[k] -= shift[k]
    theta = 0.17 + 0.64 * rep + float(row['phase'])
    if dim == 2:
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    else:
        axis = np.array([np.cos(theta) * 0.82, np.sin(theta) * 0.82, 0.572])
        axis /= np.linalg.norm(axis)
        v = np.cross([0, 0, 1], axis)
        v /= np.linalg.norm(v)
        Q = np.stack([axis, v, np.cross(axis, v)], 1)
    y = np.einsum('ji,j...->i...', Q, x)
    rho = np.sqrt(np.sum(y[1:] ** 2, axis=0))
    axis = None
    if kind in ['circle', 'sphere', 'small']:
        d = R - np.linalg.norm(y, axis=0)
        nu = y
        truth = 2 * np.pi * R if dim == 2 else 4 * np.pi * R ** 2
    elif kind in ['stadium', 'capsule']:
        d = R - np.sqrt(rho ** 2 + np.maximum(abs(y[0]) - R, 0) ** 2)
        nu = y.copy()
        nu[0] -= np.clip(y[0], -R, R)
        truth = (2 * np.pi + 4) * R if dim == 2 else 8 * np.pi * R ** 2
        axis = Q[:, 0]
    elif kind in ['ellipse', 'ellipsoid']:
        axes = np.array([1.6 * R, 0.75 * R] + ([R] if dim == 3 else []))
        d = (1 - np.sqrt(np.sum((y / axes.reshape((dim,) + (1,) * dim)) ** 2, axis=0))) * R
        nu = y / (axes ** 2).reshape((dim,) + (1,) * dim)
        axis = Q[:, 0]
        if dim == 2:
            truth = 4 * axes[0] * ellipe(1 - (axes[1] / axes[0]) ** 2)
        else:
            from numpy.polynomial.legendre import leggauss
            z, w = leggauss(120)
            ph = np.arange(240) * 2 * np.pi / 240
            a, b, c = axes
            rr = np.sqrt(1 - z[:, None] ** 2)
            truth = float((w[:, None] * np.sqrt((b * c * rr * np.cos(ph)) ** 2 + (a * c * rr * np.sin(ph)) ** 2 + (a * b * z[:, None]) ** 2)).sum() * 2 * np.pi / 240)
    elif kind == 'torus':
        r = 0.3 * R
        d = r - np.sqrt((rho - R) ** 2 + y[0] ** 2)
        nu = y.copy()
        nu[1:] *= 1 - R / np.maximum(rho, 1e-09)
        truth = 4 * np.pi ** 2 * R * r
    elif kind == 'ring':
        rr = np.linalg.norm(y, axis=0)
        r = 0.28 * R
        d = r - abs(rr - R)
        nu = y * np.sign(rr - R)
        truth = 4 * np.pi * R
    elif kind == 'multi':
        a = y.copy()
        b = y.copy()
        a[0] -= 2 * R
        b[0] += 2 * R
        ra = np.linalg.norm(a, axis=0)
        rb = np.linalg.norm(b, axis=0)
        d = R - np.minimum(ra, rb)
        nu = np.where((ra < rb)[None], a, b)
        truth = 2 * (2 * np.pi * R if dim == 2 else 4 * np.pi * R ** 2)
    else:
        raise ValueError(kind)
    normal = np.einsum('ij,j...->...i', Q, nu)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    clean = d >= 0
    mask = clean.copy()
    noise = float(row['noise'])
    valid = np.zeros_like(mask)
    valid[(slice(7, -7),) * dim] = True
    if noise:
        mask ^= (np.random.default_rng(int(row.get('noise_seed', seed))).random(mask.shape) < noise) & valid
    f = np.where(mask, 1.0, -1.0).astype('float32')
    return (f, clean, normal, d, float(truth * spacing ** (dim - 1)), axis)
