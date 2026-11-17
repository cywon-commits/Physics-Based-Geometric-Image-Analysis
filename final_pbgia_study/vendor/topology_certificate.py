import numpy as np
from itertools import permutations
def certificate(n):
    worst = np.inf
    failed = 0
    minnorm = np.inf
    for perm in permutations(range(3)):
        offsets = np.zeros((4, 3), int)
        for j, a in enumerate(perm):
            offsets[j + 1] = offsets[j]
            offsets[j + 1, a] += 1
        vs = np.stack([n[tuple((slice(o, sz - 1 + o) for sz, o in zip(n.shape[:3], off)))] for off in offsets], -2)
        center = vs.sum(-2)
        dots = np.sum(vs * center[..., None, :], -1)
        margin = dots.min(-1)
        worst = min(worst, float(margin.min()))
        bad = margin <= 1e-08
        failed += int(bad.sum())
        good = ~bad
        if good.any():
            minnorm = min(minnorm, float((margin[good] / np.linalg.norm(center[good], axis=-1)).min()))
        vbad = vs[bad].astype(float)
        if len(vbad):
            from itertools import combinations
            best = np.full(len(vbad), np.inf)
            for size in range(1, 5):
                for inds in combinations(range(4), size):
                    v = vbad[:, inds, :]
                    G = v @ v.transpose(0, 2, 1)
                    A = np.zeros((len(v), size + 1, size + 1))
                    A[:, :size, :size] = G
                    A[:, :size, size] = 1
                    A[:, size, :size] = 1
                    rhs = np.zeros(size + 1)
                    rhs[-1] = 1
                    sol = (np.linalg.pinv(A, rcond=1e-12) @ rhs)[:, :size]
                    admissible = (sol.min(1) >= -1e-08) & (abs(sol.sum(1) - 1) < 1e-06)
                    val = np.linalg.norm(np.einsum('ni,nij->nj', sol, v), axis=1)
                    best = np.minimum(best, np.where(admissible, val, np.inf))
            minnorm = min(minnorm, float(best.min()))
    return dict(hemisphere_min_dot=worst, uncertified_tetrahedra=failed, interpolation_norm_lower_bound=minnorm)
