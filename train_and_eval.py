"""
train_and_eval.py

Trains a simple pixel-level classifier (logistic regression over a small
local patch of pixels) to segment blobs, using ONLY synthetic images and
their free, perfectly-known ground-truth masks.

Then evaluates that classifier on the stand-in "real" dataset, comparing:
    (a) a model trained on the DEFAULT (untuned) simulator
    (b) a model trained on the TUNED simulator

The core claim being tested: tuning the simulator to match real, unlabeled
statistics should improve how well a model trained purely on synthetic
data generalizes to real data, even though the model never sees real
labels during training.

A logistic regression over local patches is used instead of a full CNN
to keep this dependency-light (scikit-learn + NumPy only) and fast to
run end to end. Swapping in a small CNN (e.g. a 2-layer U-Net in
PyTorch) is a natural next step once you have GPU/torch available; the
evaluation logic here would not need to change.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

from simulator import DEFAULT_PARAMS, render_batch
from target_stats import get_real_like_target, _HIDDEN_REAL_PARAMS
from tune_simulation import tune

PATCH = 2  # half-width of local patch used as features per pixel


def extract_patch_features(images):
    """
    For each pixel in each image, build a small feature vector from its
    local neighborhood. Returns (n_pixels, n_features) array.
    """
    padded = np.pad(images, ((0, 0), (PATCH, PATCH), (PATCH, PATCH)), mode="reflect")
    n, h, w = images.shape
    feats = []
    for dy in range(-PATCH, PATCH + 1):
        for dx in range(-PATCH, PATCH + 1):
            shifted = padded[:, PATCH + dy:PATCH + dy + h, PATCH + dx:PATCH + dx + w]
            feats.append(shifted.reshape(n, -1))
    return np.stack(feats, axis=-1).reshape(-1, len(feats))


def train_classifier(params, n_images=40, image_size=48, seed=1):
    images, masks = render_batch(params, n_images=n_images, image_size=image_size, seed=seed)
    X = extract_patch_features(images)
    y = masks.reshape(-1).astype(int)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X, y)
    return clf


def evaluate_on_real_like(clf, image_size=48, n_images=20, seed=999):
    """
    Evaluate a classifier on the stand-in "real" dataset. Since this is a
    prototype (no real annotated microscopy data available in this
    environment), we generate evaluation images+masks from the hidden
    real-like params purely to score IoU. In a genuine real-data setting
    you would only have a handful of real labels for validation, or you'd
    rely on visual inspection instead of an IoU number.
    """
    images, masks = render_batch(_HIDDEN_REAL_PARAMS, n_images=n_images,
                                  image_size=image_size, seed=seed)
    X = extract_patch_features(images)
    y_true = masks.reshape(-1).astype(int)
    y_pred = clf.predict(X)
    return jaccard_score(y_true, y_pred, zero_division=0)


def main():
    print("Step 1: train on DEFAULT (untuned) synthetic simulator...")
    clf_default = train_classifier(DEFAULT_PARAMS)
    iou_default = evaluate_on_real_like(clf_default)
    print(f"  IoU on real-like data (untuned simulator): {iou_default:.4f}")

    print("\nStep 2: tune simulator to match real-like unlabeled statistics...")
    tuned_params, _real_stats, _history = tune(n_iterations=60, verbose=False)
    print("  Tuned params:", {k: round(v, 3) for k, v in tuned_params.items()})

    print("\nStep 3: train on TUNED synthetic simulator...")
    clf_tuned = train_classifier(tuned_params)
    iou_tuned = evaluate_on_real_like(clf_tuned)
    print(f"  IoU on real-like data (tuned simulator):   {iou_tuned:.4f}")

    print("\n=== Summary ===")
    print(f"Untuned simulator -> real-like IoU: {iou_default:.4f}")
    print(f"Tuned simulator   -> real-like IoU: {iou_tuned:.4f}")
    change = iou_tuned - iou_default
    print(f"Change from tuning: {change:+.4f}")


if __name__ == "__main__":
    main()
