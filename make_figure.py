"""Generate a comparison figure: real-like vs default-sim vs tuned-sim images."""

import matplotlib.pyplot as plt

from simulator import DEFAULT_PARAMS, render_batch
from target_stats import _HIDDEN_REAL_PARAMS
from tune_simulation import tune


def main():
    tuned_params, real_stats, _ = tune(n_iterations=60, verbose=False)

    real_imgs, _ = render_batch(_HIDDEN_REAL_PARAMS, n_images=4, seed=42)
    default_imgs, _ = render_batch(DEFAULT_PARAMS, n_images=4, seed=42)
    tuned_imgs, _ = render_batch(tuned_params, n_images=4, seed=42)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    row_labels = ["Real-like (target)", "Default simulator", "Tuned simulator"]
    for row, imgs in enumerate([real_imgs, default_imgs, tuned_imgs]):
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=130)
    print("Saved comparison.png")


if __name__ == "__main__":
    main()
