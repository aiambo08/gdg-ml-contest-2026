# GDG on Campus UAM ML Contest: submission

- `model.py`: `class Model`. Summary features of the series and a small MLP, computed together in one compiled
  numba function per sample.
- `hf_weights.npz`: MLP weights and feature normalisation (no pickle), one set per `sample_rate`, plus a few
  reference inputs and logits used to check the compiled kernel at setup.
- `fallback_v5/`: the previous submission (a small 1-D CNN), used automatically if the fast path is unavailable.
- `requirements.txt`: `numpy` and `numba`.

Per sample: NaN interpolation, per-channel robust scale and despiking, x2 decimation when `sample_rate == 2`,
per-channel standardisation, 228 summary features (lagged cross- and auto-correlations, increment statistics at
several lags, windowed energy and energy profiles, a 48-bin spectrum of the common source of channels 0 and 1,
Teager energy, group one-hot), clipping and standardisation, MLP 228 -> 64 -> 8, argmax. The MLP was trained on
simulated test-like copies of the training set and distilled from an ensemble of 1-D CNNs. `level` is not used.

The kernel is compiled during setup (a few seconds) and checked against the stored reference logits. If numba is
not available, does not compile, or does not reproduce those logits, `model.py` loads `fallback_v5/` instead.
`predict` never raises: an unexpected failure returns -1 (abstain).
