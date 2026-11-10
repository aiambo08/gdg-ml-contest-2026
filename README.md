# GDG on Campus UAM ML Contest: submission

- `model.py`: `class Model`, a small 1-D CNN (~28k parameters).
- `weights.npz`: weights (BatchNorm folded into the convolutions), no pickle. The network has one BatchNorm
  per (`sample_rate`, `group`), so each combination has its own folded convolution weights (same cost per sample).
- `config.json`: preprocessing and architecture settings.
- `requirements.txt`: `numpy` and `numba`.

Per sample: NaN interpolation, robust per-channel z-score (median, IQR), despiking,
x2 decimation when `sample_rate == 2`, CNN, argmax.

Inference uses numba to fuse the preprocessing and the operations between layers (compiled during setup, a few
seconds). If numba is not available, or its output does not match the numpy path bit for bit at setup, the model
falls back to the pure numpy implementation, which gives identical predictions.
