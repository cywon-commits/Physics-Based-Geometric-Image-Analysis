"""Reference ONLY for 2D nonseparable 5x5 derivative basis.

Array coordinate convention: [row=y, col=x]; PyTorch conv2d uses correlation.
All six coefficients are unnormalized. Observe signs in a one-dimensional ramp test.
No training or predictions in this helper.
"""
import numpy as np

G4 = np.asarray([2.0 / 3.0, -1.0 / 12.0, 0, 0, 0, 0], dtype=float)
LEGACY = np.asarray([1., .57063894, .30414468, .33693168, .09205035, .08973684])
PILOT = np.asarray([.997721901, .262736838, .618148076, -.053713896, .265501073, -.378951811])


def derivative_kernels_2d(weights):
    """Returns (Kx, Ky) in correlation convention with +x/+y first moment.

    weights order: (abs(dy), dx) = (0,1),(0,2),(1,1),(1,2),(2,1),(2,2).
    The amplitude/first moment is deliberately NOT normalized.
    """
    w = np.asarray(weights, dtype=float)
    assert w.shape == (6,)
    kx = np.zeros((5, 5), dtype=float)
    for ay in range(3):
        for ax in (1, 2):
            coeff = w[2*ay + (ax-1)]
            yoffs = [0] if ay == 0 else [-ay, ay]
            for dy in yoffs:
                kx[2 + dy, 2 + ax] = coeff
                kx[2 + dy, 2 - ax] = -coeff
    return kx, kx.T.copy()


def first_moment_x(kx):
    return float(sum((dx-2)*kx[:, dx].sum() for dx in range(5)))


if __name__ == '__main__':
    kx, ky = derivative_kernels_2d(G4)
    assert np.allclose(kx[:, :], -kx[:, ::-1])
    assert np.allclose(kx, kx[::-1, :])
    assert np.allclose(ky, kx.T)
    assert np.isclose(first_moment_x(kx), 1.0)
    print('G4 reference 5x5 kernel and moments verified; no optimization executed')
