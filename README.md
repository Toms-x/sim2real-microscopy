# Sim-to-Real Microscopy Segmentation (in progress)

A small prototype exploring an idea from simulation-supervised machine
learning: can we **automatically tune a synthetic data simulator** so
that it matches the statistics of a real, *unlabeled* dataset, and does
that tuning actually help a model trained purely on synthetic labels
generalize better?

This is a self-directed learning project, built to get hands-on with
differentiable/optimizable simulation and sim-to-real methods.

## What's here

- `simulator.py` — a parametric generator of synthetic microscopy-like
  images: blobs (standing in for nuclei / chromatin foci) with tunable
  density, size, brightness, noise, and blur. Every image comes with a
  pixel-perfect ground-truth mask for free, since we know exactly where
  we placed each blob.
- `target_stats.py` — extracts simple, **label-free** summary statistics
  (mean intensity, contrast, coverage, blob-size proxy) from a batch of
  images. Also provides a stand-in "real" dataset (see **Honest scope**
  below).
- `tune_simulation.py` — the core piece. Uses `scipy.optimize` to search
  the simulator's parameter space so its output statistics match the
  real, unlabeled dataset's statistics.
- `train_and_eval.py` — trains a pixel classifier on synthetic labels
  only (before vs. after tuning) and evaluates both on the real-like
  data to see whether tuning improves generalization.
- `make_figure.py` — produces `comparison.png`, a visual side-by-side of
  real-like, default-simulator, and tuned-simulator images.

## Scope and limitations

This environment doesn't have access to a real annotated microscopy
dataset (e.g. the Broad Bioimage Benchmark Collection), so
`target_stats.get_real_like_target()` currently generates a stand-in
"real" dataset using a second, hidden parameter set that the tuning
code never sees directly, only its output images. The tuning code is
evaluated purely on how well it matches unlabeled pixel statistics, the
same constraint it would face with genuine real data.

**Next step:** swap `get_real_like_target()` for a loader that reads
real microscopy images (BBBC or similar), once downloaded locally.
Nothing else in the pipeline needs to change.

The optimizer here is `scipy.optimize` (a black-box search), not true
gradient descent through a differentiable renderer. It works, but
doesn't scale well to many parameters. **Next step:** rewrite
`simulator.py` in JAX or PyTorch using differentiable operations (soft
blob masks, differentiable statistics) and replace the optimizer with
Adam computed directly through the renderer.

The segmentation model is logistic regression over small local pixel
patches (scikit-learn), not a CNN. This was a deliberate choice to keep
the prototype dependency-light and fast to run end to end while
learning the overall pipeline. **Next step:** swap in a small U-Net.

## Current result (and why it's interesting)

```
Untuned simulator -> real-like IoU: 0.6838
Tuned simulator   -> real-like IoU: 0.6017
```

Tuning reduced the statistical distance between simulated and real
images by roughly 16x (0.17 -> 0.01), and `comparison.png` shows the
tuned simulator visually matches the real data's blob size and overlap
pattern far better than the default simulator does. But segmentation
IoU on the real-like data slightly *decreased* after tuning.

This is a real and useful finding, not a failed run: matching global
pixel statistics doesn't guarantee that the *task-relevant* structure
(the exact edges a segmentation model relies on) is well matched. This
is a known open problem in sim-to-real transfer, and it's exactly the
kind of gap the differentiable rendering and domain adaptation
directions in this research area are trying to close. **Next step:**
try matching task-aligned statistics instead of only pixel statistics,
for example the confidence distribution of a classifier, rather than
raw intensity and contrast.

## Running it

```bash
pip install -r requirements.txt
python simulator.py          # sanity check the generator
python tune_simulation.py    # run the statistics-matching optimization
python train_and_eval.py     # full pipeline: train, tune, retrain, compare
python make_figure.py        # generate comparison.png
```