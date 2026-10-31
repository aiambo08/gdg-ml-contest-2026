"""GDG UAM ML Contest submission: small 1-D CNN, numpy-only inference.

Files next to this one: weights.npz (BatchNorm folded into the convolutions) and config.json.
Pipeline per sample: NaN interpolation -> robust z (median, IQR/1.349) per channel -> despike (|z|>k ->
re-interpolate, clip) -> sr=2 decimated x2 (mode 'dec') -> CNN -> argmax.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from competition_api import Sample  # noqa: F401
except Exception:  # pragma: no cover
    Sample = object

HERE = Path(__file__).resolve().parent
_AR = np.arange(256)


def _fill_nan(x, nanm):
    """Vectorised linear interpolation of NaN along axis 1 (same result as np.interp: ends take the
    nearest valid value, an all-NaN row becomes 0). x (C,T) float32, modified in place."""
    C, T = x.shape
    ar = _AR[:T]
    prev = np.maximum.accumulate(np.where(nanm, -1, ar), axis=1)
    nxt = np.minimum.accumulate(np.where(nanm, T, ar)[:, ::-1], axis=1)[:, ::-1]
    r, c = np.nonzero(nanm)
    p = prev[r, c]
    n = nxt[r, c]
    pc = np.maximum(p, 0)
    nc = np.minimum(n, T - 1)
    xp = x[r, pc]
    xn = x[r, nc]
    den = np.maximum(n - p, 1)
    v = xp + (xn - xp) * ((c - p) / den).astype(np.float32)
    v = np.where(p < 0, xn, v)
    v = np.where(n >= T, xp, v)
    v = np.where((p < 0) & (n >= T), 0.0, v)
    x[r, c] = v
    return x


def median3(x):
    a = np.empty_like(x); c = np.empty_like(x)
    a[:, 1:] = x[:, :-1]; a[:, 0] = x[:, 0]
    c[:, :-1] = x[:, 1:]; c[:, -1] = x[:, -1]
    return a + x + c - np.maximum(np.maximum(a, x), c) - np.minimum(np.minimum(a, x), c)


def preprocess(series, sr, channels, mode, k, med3=False):
    x = np.array(series if channels is None else series[channels], np.float32, order="C")   # (C,256) copy
    nanm = ~np.isfinite(x)                    # NaN and +-inf are both treated as missing
    if nanm.any():
        _fill_nan(x, nanm)
    s = np.sort(x, axis=1)                    # exact np.percentile (linear) at 25/50/75 for T=256
    q1 = s[:, 63] + 0.75 * (s[:, 64] - s[:, 63])
    med = 0.5 * (s[:, 127] + s[:, 128])
    q3 = s[:, 191] + 0.25 * (s[:, 192] - s[:, 191])
    rs = np.maximum((q3 - q1) / np.float32(1.349), np.float32(1e-3))
    x -= med[:, None]
    x /= rs[:, None]
    spk = np.abs(x) > k
    if spk.any():
        _fill_nan(x, spk)
    np.clip(x, -k, k, out=x)
    if med3:
        x = median3(x)
    if mode == "dec" and sr == 2:
        x = x.reshape(x.shape[0], -1, 2).mean(-1)
    return x


class Model:
    def __init__(self):
        cfg = json.loads((HERE / "config.json").read_text())
        self.ch = None if list(cfg["channels"]) == [0, 1, 2, 3] else list(cfg["channels"])
        self.mode = cfg["mode"]
        self.k = float(cfg["k"])
        self.med3 = bool(cfg.get("med3", 0))
        self.use_sr = bool(cfg["use_sr"])
        w = np.load(HERE / "weights.npz")
        n = cfg["n_layers"]
        # conv weights as (Cin*k, Cout) for (L, Cin*k) @ W ; activations kept as (L, C)
        self.layers = []
        for i in range(n):
            W = w[f"w{i}"]                       # (Cout, Cin, k)
            cout, cin, kk = W.shape
            Wk = W.transpose(0, 2, 1).reshape(cout, kk * cin).T          # rows ordered (tap, cin)
            self.layers.append((np.ascontiguousarray(Wk), w[f"b{i}"].astype(np.float32), kk))
        self.hw = np.ascontiguousarray(w["hw"].T)
        self.hb = w["hb"]

    def _forward_np(self, x, sr):
        h = x.T                                   # (L, C)
        n = len(self.layers)
        for i, (W, b, kk) in enumerate(self.layers):
            L, C = h.shape
            p = kk // 2
            hp = np.zeros((L + 2 * p, C), np.float32)
            hp[p:p + L] = h
            cols = np.concatenate([hp[j:j + L] for j in range(kk)], axis=1)   # (L, k*C)
            h = cols @ W
            h += b
            np.maximum(h, 0, out=h)
            if i < n - 1:
                L2 = L // 2
                h = h[:2 * L2].reshape(L2, 2, -1).max(1)
        f = np.concatenate([h.mean(0), h.max(0)])
        if self.use_sr:
            f = np.append(f, np.float32(sr - 1.5))
        return f @ self.hw + self.hb

    def logits(self, x) -> np.ndarray:
        try:
            sr = int(round(float(x.meta.get("sample_rate", 1))))
        except (TypeError, ValueError):
            sr = 1
        z = preprocess(x.series, sr, self.ch, self.mode, self.k, self.med3)
        return self._forward_np(z, sr)

    def predict(self, x) -> int:
        # Any failure abstains: an abstention scores better than a sure wrong answer.
        try:
            out = self.logits(x)
        except Exception:
            return -1
        if not np.isfinite(out).all():
            return -1
        return int(np.argmax(out))
