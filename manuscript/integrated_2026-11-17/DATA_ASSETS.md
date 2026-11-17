# Packaged numerical arrays

The four `.npz` files store a NumPy array under the key `data` (load via `np.load(path)['data']`): the synthetic strata analytic ground-truth angles, plus CC0 example image arrays for the brick, grass and retina demonstrations.

Some historical scripts regenerate the equivalent `.npy` files using `np.save`. These compressed copies are included to keep the GitHub manuscript bundle compact; to make an `.npy` if needed, run `np.save('name.npy', np.load('name.npz')['data'])`.

The manuscript `main.pdf` is a 24-page screen-resolution preview. The TeX build uses the original vector PDF figures in `figures/` and generates a full-resolution PDF.
