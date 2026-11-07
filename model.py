"""GDG UAM ML Contest submission: small 1-D CNN, numpy-only inference (v2, latency-optimised).

Same weights and same float32 arithmetic as v1 (logits are bit-identical); the per-sample work is
reorganised to make far fewer numpy calls:
- gap filling works on the few masked runs (typically ~4 per sample) instead of the whole (C, T) array;
- preallocated, zero-padded activation buffers per (sample_rate, layer); the im2col matrix is a fixed
  strided view of the buffer, so each conv is a single matmul into a preallocated output;
- max-pool is done before bias and ReLU (exactly equivalent: rounding is monotonic) and writes straight
  into the next layer's buffer;
- cheap min/max/sum checks skip the NaN mask and the despike work when they cannot change anything.

Files next to this one: weights.npz (BatchNorm folded into the convolutions) and config.json.
Pipeline per sample: NaN interpolation -> robust z (median, IQR/1.349) per channel -> despike (|z|>k ->
re-interpolate, clip) -> sr=2 decimated x2 (mode 'dec') -> CNN -> argmax.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

try:
    from competition_api import Sample  # noqa: F401
except Exception:  # pragma: no cover
    Sample = object

HERE = Path(__file__).resolve().parent
_IQR = np.float32(1.349)
_EPS = np.float32(1e-3)
_Q75 = np.float32(0.75)
_Q25 = np.float32(0.25)
_HALF = np.float32(0.5)
_LOHI = (63, 191)                      # linear-interpolated quartiles at 25 % and 75 % for T = 256


def _fill_runs(x, pos):
    """Linear interpolation of the entries at flat positions `pos` (sorted) of x (C, T), row by row, in
    place. Same float32 arithmetic as v1's vectorised version (and np.interp's rule): interior runs take
    xp + (xn - xp) * f32((c - p) / (n - p)); runs touching an end take the nearest valid value; a fully
    masked row becomes 0."""
    T = x.shape[1]
    xf = x.reshape(-1)                 # view (x is C-contiguous)
    P = pos.tolist()
    m = len(P)
    ipos, ipc, inc, num, den = [], [], [], [], []
    i = 0
    while i < m:
        s = P[i]
        r = s // T
        j = i
        lim = r * T + T - 1
        while j + 1 < m and P[j + 1] == P[j] + 1 and P[j] < lim:
            j += 1
        e = P[j]
        base = r * T
        p = s - base - 1               # previous valid column (-1: none)
        q = e - base + 1               # next valid column (T: none)
        if p < 0 and q >= T:
            xf[s:e + 1] = 0.0
        elif p < 0:
            xf[s:e + 1] = xf[base + q]
        elif q >= T:
            xf[s:e + 1] = xf[base + p]
        else:
            d = q - p
            for c in range(s - base, e - base + 1):
                ipos.append(base + c); ipc.append(base + p); inc.append(base + q)
                num.append(c - p); den.append(d)
        i = j + 1
    if ipos:
        xp = xf[ipc]
        xn = xf[inc]
        t = (np.array(num) / np.array(den)).astype(np.float32)
        xf[ipos] = xp + (xn - xp) * t
    return x


def median3(x):
    a = np.empty_like(x); c = np.empty_like(x)
    a[:, 1:] = x[:, :-1]; a[:, 0] = x[:, 0]
    c[:, :-1] = x[:, 1:]; c[:, -1] = x[:, -1]
    return a + x + c - np.maximum(np.maximum(a, x), c) - np.minimum(np.minimum(a, x), c)


class _Plan:
    """Preallocated buffers and fixed views for one input length (one per sample_rate)."""

    def __init__(self, layers, cin0, L0):
        self.steps = []
        L, C = L0, cin0
        n = len(layers)
        ins = []
        for i, (W, b, kk) in enumerate(layers):
            p = kk // 2
            buf = np.zeros((L + 2 * p, C), np.float32)
            s0, s1 = buf.strides
            # (L, kk, C) strided view -> (L, kk*C) rows ordered (tap, cin), as the weights expect.
            # Merging (kk, C) is stride-compatible, so this stays a view of buf (no copy per call).
            cols = as_strided(buf, (L, kk, C), (s0, s0, s1), writeable=False).reshape(L, kk * C)
            assert np.shares_memory(cols, buf)
            ins.append(buf[p:p + L])
            cout = W.shape[1]
            out = np.empty((L, cout), np.float32)
            self.steps.append([cols, W, b, out, None, None, None])
            C = cout
            if i < n - 1:
                L = L // 2
        for i in range(n - 1):
            st = self.steps[i]
            out = st[3]
            L2 = ins[i + 1].shape[0]
            st[4], st[5], st[6] = out[0:2 * L2:2], out[1:2 * L2:2], ins[i + 1]
        self.h0 = ins[0]
        self.last = self.steps[-1]
        self.body = self.steps[:-1]
        self.Lout = L
        self.feat = np.empty(2 * C, np.float32)
        self.fmean, self.fmax = self.feat[:C], self.feat[C:]


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
        self.layers = []
        for i in range(n):
            W = w[f"w{i}"]                       # (Cout, Cin, k)
            cout, cin, kk = W.shape
            Wk = W.transpose(0, 2, 1).reshape(cout, kk * cin).T          # rows ordered (tap, cin)
            self.layers.append((np.ascontiguousarray(Wk, np.float32), w[f"b{i}"].astype(np.float32), kk))
        self.hw = np.ascontiguousarray(w["hw"].T)
        self.hb = w["hb"]
        cin0 = 4 if self.ch is None else len(self.ch)
        self._plans = {1: _Plan(self.layers, cin0, 256)}
        if self.mode == "dec":
            self._plans[2] = _Plan(self.layers, cin0, 128)

    # ---------------------------------------------------------------- preprocessing, (C, T) layout
    def _prep(self, series):
        x = np.array(series if self.ch is None else series[self.ch], np.float32, order="C")  # (C,256)
        if not math.isfinite(float(x.sum())):    # NaN / +-inf (or a harmless overflow: mask is exact)
            pos = np.flatnonzero(~np.isfinite(x))
            if len(pos):
                _fill_runs(x, pos)
        s = np.sort(x, axis=1)                    # exact np.percentile (linear) at 25/50/75 for T=256
        q1 = s[:, 63] + _Q75 * (s[:, 64] - s[:, 63])
        med = _HALF * (s[:, 127] + s[:, 128])
        q3 = s[:, 191] + _Q25 * (s[:, 192] - s[:, 191])
        rs = np.maximum((q3 - q1) / _IQR, _EPS)
        x -= med[:, None]
        x /= rs[:, None]
        k = self.k
        if not (-k <= x.min() and x.max() <= k):  # also taken when x holds NaN: the mask is exact
            pos = np.flatnonzero(np.abs(x) > k)
            if len(pos):
                _fill_runs(x, pos)
            np.clip(x, -k, k, out=x)              # a no-op whenever this branch is skipped
        if self.med3:
            x = median3(x)
        return x

    def logits(self, x) -> np.ndarray:
        try:
            sr = int(round(float(x.meta.get("sample_rate", 1))))
        except (TypeError, ValueError):
            sr = 1
        z = self._prep(x.series)
        if self.mode == "dec" and sr == 2:        # mean of pairs == (a + b) * 0.5 exactly
            plan = self._plans[2]
            zt = z.T
            h0 = plan.h0
            np.add(zt[0::2], zt[1::2], out=h0)
            h0 *= _HALF
        else:
            plan = self._plans[1]
            np.copyto(plan.h0, z.T)
        mm, mx, add = np.matmul, np.maximum, np.add
        for cols, W, b, out, ev, od, nxt in plan.body:
            mm(cols, W, out=out)
            mx(ev, od, out=nxt)                   # max-pool, then bias + ReLU (identical to v1's order)
            add(nxt, b, out=nxt)
            mx(nxt, 0, out=nxt)
        cols, W, b, h = plan.last[:4]
        mm(cols, W, out=h)
        add(h, b, out=h)
        mx(h, 0, out=h)
        add.reduce(h, axis=0, out=plan.fmean)     # == h.mean(0): same pairwise sum, then / L
        np.true_divide(plan.fmean, plan.Lout, out=plan.fmean)
        mx.reduce(h, axis=0, out=plan.fmax)
        f = plan.feat
        if self.use_sr:
            f = np.append(f, np.float32(sr - 1.5))
        return f @ self.hw + self.hb

    def predict(self, x) -> int:
        # Any failure abstains: an abstention scores better than a sure wrong answer.
        try:
            out = self.logits(x)
            if not math.isfinite(float(out.sum())):
                return -1
            return int(out.argmax())
        except Exception:
            return -1
