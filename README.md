# GDG on Campus UAM ML Contest: submission

- `model.py`: `class Model`, a small 1-D CNN (26k parameters) with numpy-only inference.
- `weights.npz`: weights (BatchNorm folded into the convolutions), no pickle.
- `config.json`: preprocessing and architecture settings.
- `requirements.txt`: `numpy` only.

Per sample: NaN interpolation, robust per-channel z-score (median, IQR), despiking,
x2 decimation when `sample_rate == 2`, CNN, argmax.
