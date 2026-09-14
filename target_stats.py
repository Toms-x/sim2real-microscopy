"""
target_stats.py

Two responsibilities:

1. compute_stats(images) - extract simple, unlabeled-friendly summary
   statistics from a batch of images (mean intensity, contrast/std,
   estimated blob density via peak counting, estimated blob size via
   autocorrelation width). These are the numbers our simulator tries
   to match; they require no labels, only images, which is what makes
   this approach usable on real unlabeled microscopy data.

2. get_real_like_target(...) - stands in for a real, unlabeled dataset.

   IMPORTANT: in this prototype there is no internet access to actual
   biological microscopy data (e.g. the Broad Bioimage Benchmark
   Collection), so this function generates a batch using a DIFFERENT,
   hidden parameter set that the tuning code never sees directly. The
   tuning code only ever gets to look at this batch's pixel statistics,
   never its underlying parameters, exactly like it would with a real
   unlabeled dataset. Swap this function out for a real image loader
   once you have downloaded actual microscopy data (see README).
"""

import numpy as np
from simulator import render_batch

# Hidden "ground truth" params for the stand-in real dataset.
# The tuning code in tune_simulation.py is NOT allowed to read this dict.
_HIDDEN_REAL_PARAMS = {
    "density": 18.0,
    "mean_radius": 5.5,
    "radius_std": 1.6,
    "intensity": 0.85,
    "noise_level": 0.12,
    "blur_sigma": 1.6,
}


def get_real_like_target(n_images=32, image_size=64, seed=123):
    """Return a batch of images standing in for unlabeled real data."""
    images, _masks = render_batch(_HIDDEN_REAL_PARAMS, n_images=n_images,
                                   image_size=image_size, seed=seed)
    return images  # masks intentionally discarded: real data has no labels


def compute_stats(images):
    """
    Compute simple, label-free summary statistics from a batch of images.
    Returns a dict of scalar statistics.
    """
    mean_intensity = images.mean()
    contrast = images.std()

    # Rough blob-density proxy: fraction of pixels above an adaptive threshold
    thresh = mean_intensity + 0.5 * contrast
    coverage = (images > thresh).mean()

    # Rough blob-size proxy: average size of a local bright patch, estimated
    # via the autocorrelation width of the thresholded image.
    sizes = []
    for img in images:
        binary = (img > thresh).astype(np.float64)
        if binary.sum() == 0:
            continue
        f = np.fft.fft2(binary)
        autocorr = np.fft.ifft2(f * np.conj(f)).real
        autocorr = np.fft.fftshift(autocorr)
        center = autocorr[autocorr.shape[0] // 2, autocorr.shape[1] // 2]
        if center > 0:
            half = autocorr.shape[0] // 2
            row = autocorr[half, half:] / center
            width = np.searchsorted(row < 0.5, True)
            sizes.append(max(width, 1))
    size_proxy = float(np.mean(sizes)) if sizes else 0.0

    return {
        "mean_intensity": float(mean_intensity),
        "contrast": float(contrast),
        "coverage": float(coverage),
        "size_proxy": size_proxy,
    }


if __name__ == "__main__":
    real_like = get_real_like_target(n_images=16)
    stats = compute_stats(real_like)
    print("Stand-in 'real' stats:", stats)
