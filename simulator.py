"""
simulator.py

A parametric, tunable simulator that generates synthetic microscopy-like
images of blobs (standing in for cell nuclei / chromatin foci).

The simulator has a small set of interpretable parameters:
    density       - expected number of blobs per image
    mean_radius   - average blob radius (pixels)
    radius_std    - spread of blob radii
    intensity     - peak brightness of a blob
    noise_level   - standard deviation of additive Gaussian noise
    blur_sigma    - Gaussian blur applied to the whole image (optical blur)

Because every image is generated from known blob positions and radii,
we get pixel-perfect segmentation masks for free (no manual annotation).

This is deliberately framework-light (NumPy only) so it runs anywhere.
The same parameters and statistics-matching idea carry over directly to
a JAX/PyTorch version where the renderer is written with autodiff ops
instead of NumPy ops (see README for the upgrade path).
"""

import numpy as np


DEFAULT_PARAMS = {
    "density": 12.0,       # expected blob count
    "mean_radius": 4.0,
    "radius_std": 1.0,
    "intensity": 1.0,
    "noise_level": 0.05,
    "blur_sigma": 1.0,
}


def _gaussian_blur(img, sigma):
    """Simple separable Gaussian blur, implemented with NumPy only."""
    if sigma <= 0:
        return img
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    # convolve rows then columns
    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=img)
    img = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=img)
    return img


def render_image(params, image_size=64, rng=None):
    """
    Render one synthetic image + its ground-truth binary mask.

    Returns:
        image: (H, W) float array in [0, 1]
        mask:  (H, W) binary array, 1 where a blob covers the pixel
    """
    rng = rng or np.random.default_rng()

    n_blobs = max(0, int(round(rng.poisson(params["density"]))))
    image = np.zeros((image_size, image_size), dtype=np.float64)
    mask = np.zeros((image_size, image_size), dtype=np.float64)

    yy, xx = np.mgrid[0:image_size, 0:image_size]

    for _ in range(n_blobs):
        cx = rng.uniform(0, image_size)
        cy = rng.uniform(0, image_size)
        r = max(0.5, rng.normal(params["mean_radius"], params["radius_std"]))
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = params["intensity"] * np.exp(-dist_sq / (2 * r ** 2))
        image += blob
        mask[dist_sq <= r ** 2] = 1.0

    image = _gaussian_blur(image, params["blur_sigma"])
    image += rng.normal(0, params["noise_level"], size=image.shape)
    image = np.clip(image, 0, 1)

    return image, mask


def render_batch(params, n_images=32, image_size=64, seed=None):
    """Render a batch of (image, mask) pairs using a fixed simulator config."""
    rng = np.random.default_rng(seed)
    images, masks = [], []
    for _ in range(n_images):
        img, msk = render_image(params, image_size=image_size, rng=rng)
        images.append(img)
        masks.append(msk)
    return np.stack(images), np.stack(masks)


if __name__ == "__main__":
    imgs, masks = render_batch(DEFAULT_PARAMS, n_images=4, seed=0)
    print("Rendered batch:", imgs.shape, masks.shape)
    print("Mean intensity:", imgs.mean(), "| Mean mask coverage:", masks.mean())
