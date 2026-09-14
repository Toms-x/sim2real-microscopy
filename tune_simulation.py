"""
tune_simulation.py

Tunes the simulator's parameters so that images it generates match the
pixel statistics of an unlabeled "real" dataset, without ever looking at
the real dataset's underlying ground-truth parameters or labels.

This is the core idea from the project brief: automatically tune a
simulation to be realistic enough that models trained on it generalize
to real data, by matching statistics of the real, unlabeled distribution.

Method used here: scipy.optimize.minimize (Nelder-Mead) treats the
simulator as a black box and searches parameter space to minimize the
distance between simulated and real statistics. This works without
needing autodiff, at the cost of being less sample-efficient than a
gradient-based method.

Upgrade path: rewrite render_image/render_batch in JAX or PyTorch using
differentiable operations (soft blob masks instead of hard thresholds,
differentiable statistics), then replace this optimizer with
gradient descent (Adam) computed directly through the renderer. That
removes the black-box search and lets the tuning scale to many more
parameters, which is the direction described in the PhD project brief.
"""

import numpy as np
from scipy.optimize import minimize

from simulator import DEFAULT_PARAMS, render_batch
from target_stats import get_real_like_target, compute_stats

PARAM_NAMES = ["density", "mean_radius", "radius_std", "intensity",
               "noise_level", "blur_sigma"]


def params_to_vector(params):
    return np.array([params[name] for name in PARAM_NAMES])


def vector_to_params(vec):
    vec = np.clip(vec, 1e-3, None)  # keep all params positive
    return dict(zip(PARAM_NAMES, vec))


def stats_distance(sim_stats, real_stats):
    """Normalized squared distance between two stats dicts."""
    keys = sim_stats.keys()
    diffs = []
    for k in keys:
        scale = max(abs(real_stats[k]), 1e-3)
        diffs.append(((sim_stats[k] - real_stats[k]) / scale) ** 2)
    return float(np.mean(diffs))


def make_objective(real_stats, n_images=24, image_size=64, seed=0):
    def objective(vec):
        params = vector_to_params(vec)
        images, _masks = render_batch(params, n_images=n_images,
                                       image_size=image_size, seed=seed)
        sim_stats = compute_stats(images)
        return stats_distance(sim_stats, real_stats)
    return objective


def tune(n_iterations=60, verbose=True):
    real_images = get_real_like_target(n_images=32)
    real_stats = compute_stats(real_images)

    x0 = params_to_vector(DEFAULT_PARAMS)
    objective = make_objective(real_stats)

    history = []

    def callback(vec):
        history.append(objective(vec))

    result = minimize(
        objective, x0, method="Nelder-Mead",
        options={"maxiter": n_iterations, "xatol": 1e-3, "fatol": 1e-4},
        callback=callback,
    )

    tuned_params = vector_to_params(result.x)

    if verbose:
        start_loss = objective(x0)
        end_loss = objective(result.x)
        print("Real (target) stats:      ", {k: round(v, 4) for k, v in real_stats.items()})
        print("Default sim params:       ", DEFAULT_PARAMS)
        print("Default sim stats:        ",
              {k: round(v, 4) for k, v in compute_stats(render_batch(DEFAULT_PARAMS, 32)[0]).items()})
        print(f"Distance before tuning:    {start_loss:.4f}")
        print("Tuned sim params:         ", {k: round(v, 3) for k, v in tuned_params.items()})
        print("Tuned sim stats:          ",
              {k: round(v, 4) for k, v in compute_stats(render_batch(tuned_params, 32)[0]).items()})
        print(f"Distance after tuning:     {end_loss:.4f}")

    return tuned_params, real_stats, history


if __name__ == "__main__":
    tune()
