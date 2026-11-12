"""GDG UAM ML Contest submission (v4): small 1-D CNN, numba-fused inference with a numpy fallback.

Model = the numpy model (_ModelNP below, unchanged) whose logits() is replaced, when numba imports,
compiles and reproduces the numpy logits bit for bit at setup, by a path with ~12 Python-level calls per
sample instead of ~45:
- one numba function does NaN filling, robust z, despike, decimation and writes the layer-1 im2col matrix;
- after each matmul, one numba function does max-pool + bias + ReLU and writes the next im2col matrix;
- the convolutions stay numpy matmuls (BLAS), now on contiguous im2col matrices.
Otherwise (no numba, or any mismatch at setup) it runs the numpy path unchanged.

Numpy model description:

Same weights and same float32 arithmetic as v1 (logits are bit-identical); the per-sample work is
reorganised to make far fewer numpy calls:
- gap filling works on the few masked runs (typically ~4 per sample) instead of the whole (C, T) array;
- preallocated, zero-padded activation buffers per (sample_rate, layer); the im2col matrix is a fixed
  strided view of the buffer, so each conv is a single matmul into a preallocated output;
- max-pool is done before bias and ReLU (exactly equivalent: rounding is monotonic) and writes straight
  into the next layer's buffer;
- cheap min/max/sum checks skip the NaN mask and the despike work when they cannot change anything.

Files next to this one: weights.npz (BatchNorm folded into the convolutions) and config.json.
If config.json has "srbn": true, the network has one BatchNorm per sample_rate, so each sr has its own folded
conv weights (w{i}_s1 / w{i}_s2); with "gbn": true, one per (sample_rate, group): w{i}_s{1,2}_g{0,1,2} for
groups A, B, C. The work per sample is unchanged: only the weight set is chosen from the metadata.
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
_GROUPS = {"A": 0, "B": 1, "C": 2}
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


class _ModelNP:
    def __init__(self):
        cfg = json.loads((HERE / "config.json").read_text())
        self.ch = None if list(cfg["channels"]) == [0, 1, 2, 3] else list(cfg["channels"])
        self.mode = cfg["mode"]
        self.k = float(cfg["k"])
        self.med3 = bool(cfg.get("med3", 0))
        self.use_sr = bool(cfg["use_sr"])
        w = np.load(HERE / "weights.npz")
        n = cfg["n_layers"]
        srbn = bool(cfg.get("srbn", False))      # one folded conv set per sample_rate (BN per sr)

        def layers_of(sfx):
            out = []
            for i in range(n):
                W = w[f"w{i}{sfx}"]                  # (Cout, Cin, k)
                cout, cin, kk = W.shape
                Wk = W.transpose(0, 2, 1).reshape(cout, kk * cin).T      # rows ordered (tap, cin)
                out.append((np.ascontiguousarray(Wk, np.float32), w[f"b{i}{sfx}"].astype(np.float32), kk))
            return out

        gbn = bool(cfg.get("gbn", False))        # one folded conv set per (sample_rate, group)
        self.ngroups = 3 if gbn else 1
        self._layers_by = {}
        for g in range(self.ngroups):
            for sr in (1, 2):
                sfx = f"_s{sr}_g{g}" if gbn else (f"_s{sr}" if srbn else "")
                self._layers_by[(sr, g)] = layers_of(sfx)
        self.layers = self._layers_by[(1, 0)]
        self.hw = np.ascontiguousarray(w["hw"].T)
        self.hb = w["hb"]
        cin0 = 4 if self.ch is None else len(self.ch)
        self._plans = {}
        for (sr, g), lay in self._layers_by.items():
            if sr == 1 or self.mode == "dec":
                self._plans[(sr, g)] = _Plan(lay, cin0, 256 if sr == 1 else 128)

    def _group(self, meta):
        """Group index for the (sr, group) weight sets; a missing or unknown group uses B (middle gain)."""
        if self.ngroups == 1:
            return 0
        try:
            return _GROUPS.get(str(meta.get("group", "B")).strip().upper(), 1)
        except Exception:
            return 1

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
        g = self._group(x.meta)
        z = self._prep(x.series)
        if self.mode == "dec" and sr == 2:        # mean of pairs == (a + b) * 0.5 exactly
            plan = self._plans[(2, g)]
            zt = z.T
            h0 = plan.h0
            np.add(zt[0::2], zt[1::2], out=h0)
            h0 *= _HALF
        else:
            plan = self._plans[(1, g)]
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


# ====================================================================== numba path
try:
    import numba as _nb
    _njit = _nb.njit(cache=False, fastmath=False, nogil=True)
except Exception:  # numba missing or broken -> numpy path
    _nb = None

if _nb is not None:
    _F0 = np.float32(0.0)

    @_njit
    def _nb_fill(x, mask):
        C, T = x.shape
        for r in range(C):
            j = 0
            while j < T:
                if not mask[r, j]:
                    j += 1
                    continue
                s = j
                while j < T and mask[r, j]:
                    j += 1
                e = j - 1
                p = s - 1
                q = e + 1
                if p < 0 and q >= T:
                    for c in range(s, e + 1):
                        x[r, c] = _F0
                elif p < 0:
                    v = x[r, q]
                    for c in range(s, e + 1):
                        x[r, c] = v
                elif q >= T:
                    v = x[r, p]
                    for c in range(s, e + 1):
                        x[r, c] = v
                else:
                    xp = x[r, p]
                    dx = x[r, q] - xp
                    d = q - p
                    for c in range(s, e + 1):
                        t = np.float32((c - p) / d)
                        x[r, c] = xp + dx * t

    @_njit
    def _nb_im2col(buf, cols, kk):
        L = cols.shape[0]
        C = buf.shape[1]
        for l in range(L):
            for t in range(kk):
                for c in range(C):
                    cols[l, t * C + c] = buf[l + t, c]

    @_njit
    def _nb_prep(src, x, mask, sbuf, k, dec, buf, p, cols, kk):
        C, T = x.shape
        bad = False
        for r in range(C):
            for c in range(T):
                v = np.float32(src[r, c])
                x[r, c] = v
                b = not np.isfinite(v)
                mask[r, c] = b
                bad = bad or b
        if bad:
            _nb_fill(x, mask)
        mn = np.float32(np.inf)
        mx = np.float32(-np.inf)
        anynan = False
        for r in range(C):
            for c in range(T):
                sbuf[c] = x[r, c]
            sbuf.sort()
            q1 = sbuf[63] + np.float32(0.75) * (sbuf[64] - sbuf[63])
            med = np.float32(0.5) * (sbuf[127] + sbuf[128])
            q3 = sbuf[191] + np.float32(0.25) * (sbuf[192] - sbuf[191])
            rs = (q3 - q1) / np.float32(1.349)
            if rs != rs:
                pass                               # np.maximum propagates NaN
            elif rs < np.float32(1e-3):
                rs = np.float32(1e-3)
            for c in range(T):
                v = (x[r, c] - med) / rs
                x[r, c] = v
                if v != v:
                    anynan = True
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        if anynan or not (-k <= mn and mx <= k):
            spk = False
            for r in range(C):
                for c in range(T):
                    b = abs(x[r, c]) > k
                    mask[r, c] = b
                    spk = spk or b
            if spk:
                _nb_fill(x, mask)
            kf = np.float32(k)
            for r in range(C):
                for c in range(T):
                    v = x[r, c]
                    if v < -kf:
                        x[r, c] = -kf
                    elif v > kf:
                        x[r, c] = kf
        L = buf.shape[0] - 2 * p
        if dec:
            h = np.float32(0.5)
            for i in range(L):
                for r in range(C):
                    buf[p + i, r] = (x[r, 2 * i] + x[r, 2 * i + 1]) * h
        else:
            for i in range(L):
                for r in range(C):
                    buf[p + i, r] = x[r, i]
        _nb_im2col(buf, cols, kk)

    @_njit
    def _nb_post(out, b, buf, p, cols, kk):
        L2 = buf.shape[0] - 2 * p
        C = out.shape[1]
        for i in range(L2):
            for c in range(C):
                a = out[2 * i, c]
                o = out[2 * i + 1, c]
                v = (a if a >= o else o) + b[c]
                buf[p + i, c] = v if v > _F0 else _F0
        _nb_im2col(buf, cols, kk)

    @_njit
    def _nb_last(h, b, fmean, fmax):
        L, C = h.shape
        for i in range(L):
            for c in range(C):
                v = h[i, c] + b[c]
                v = v if v > _F0 else _F0
                if i == 0:
                    fmean[c] = v
                    fmax[c] = v
                else:
                    fmean[c] += v
                    if v > fmax[c]:
                        fmax[c] = v
        for c in range(C):
            fmean[c] = fmean[c] / np.float32(L)


class _NbPlan:
    """Contiguous padded buffers, im2col matrices and outputs for one sample_rate."""

    def __init__(self, layers, cin0, L0):
        self.layers = []
        L, C = L0, cin0
        n = len(layers)
        for i, (W, b, kk) in enumerate(layers):
            p = kk // 2
            buf = np.zeros((L + 2 * p, C), np.float32)
            cols = np.empty((L, kk * C), np.float32)
            out = np.empty((L, W.shape[1]), np.float32)
            self.layers.append((buf, p, cols, kk, W, b, out))
            C = W.shape[1]
            if i < n - 1:
                L //= 2
        self.feat = np.empty(2 * C, np.float32)
        self.fmean, self.fmax = self.feat[:C], self.feat[C:]


class _Smp:
    def __init__(self, series, sr, grp):
        self.series = series
        self.meta = {"sample_rate": sr, "group": grp}


class Model(_ModelNP):
    def __init__(self):
        super().__init__()
        self._nb_ok = False
        try:
            if _nb is None or self.ch is not None or self.med3 or self.use_sr or self.mode != "dec":
                return
            self._nbp = {(sr, g): _NbPlan(lay, 4, 256 if sr == 1 else 128) for (sr, g), lay in self._layers_by.items()}
            self._x = np.empty((4, 256), np.float32)
            self._mask = np.empty((4, 256), np.bool_)
            self._sbuf = np.empty(256, np.float32)
            # compile, then check bit-identity against the numpy path (both sample rates, NaN, gaps, spikes)
            rng = np.random.default_rng(0)
            for sr, grp in ((1, "A"), (2, "A"), (1, "B"), (2, "B"), (1, "C"), (2, "C"), (2, "?"), (1, "A")):
                s = rng.standard_normal((4, 256)).astype(np.float32)
                s[rng.random((4, 256)) < 0.05] = np.nan
                s[1, 30:45] = np.nan
                s[2, :7] = np.nan
                s[0, 10] = 1e4
                smp = _Smp(s, sr, grp)
                if not np.array_equal(self._logits_nb(smp), _ModelNP.logits(self, smp)):
                    return
            self._nb_ok = True
        except Exception:
            self._nb_ok = False

    def _logits_nb(self, x):
        try:
            sr = int(round(float(x.meta.get("sample_rate", 1))))
        except (TypeError, ValueError):
            sr = 1
        dec = sr == 2
        P = self._nbp[(2 if dec else 1, self._group(x.meta))]
        Ls = P.layers
        buf, p, cols, kk = Ls[0][:4]
        # numba compiles one specialisation per argument type, and a read-only array (what the harness
        # passes: np.frombuffer of bytes), another dtype or another layout are all different types: the
        # first such call would recompile _nb_prep (~1-3 s) inside predict. So the series is first copied
        # into our own float32 buffer and numba only ever sees the type compiled at setup.
        s = x.series
        if getattr(s, "shape", None) != (4, 256):
            return _ModelNP.logits(self, x)
        np.copyto(self._x, s, casting="unsafe")
        _nb_prep(self._x, self._x, self._mask, self._sbuf, self.k, dec, buf, p, cols, kk)
        mm = np.matmul
        n = len(Ls)
        for i in range(n - 1):
            cols, W, b, out = Ls[i][2], Ls[i][4], Ls[i][5], Ls[i][6]
            mm(cols, W, out=out)
            nbuf, np_, ncols, nkk = Ls[i + 1][:4]
            _nb_post(out, b, nbuf, np_, ncols, nkk)
        cols, W, b, out = Ls[n - 1][2], Ls[n - 1][4], Ls[n - 1][5], Ls[n - 1][6]
        mm(cols, W, out=out)
        _nb_last(out, b, P.fmean, P.fmax)
        return P.feat @ self.hw + self.hb

    def logits(self, x):
        if self._nb_ok:
            return self._logits_nb(x)
        return _ModelNP.logits(self, x)
