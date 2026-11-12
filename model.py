"""GDG UAM ML Contest submission (v6): summary features + a small MLP, fused in one numba kernel.

Fast path (one compiled call per sample, no allocations):
  NaN/inf interpolation -> winsorised scale per channel -> despike (|z| > 8 re-interpolated, clipped) ->
  sample_rate 2 decimated x2 (physical time) -> per-channel standardisation -> summary features
  (lagged cross/auto-correlations, increment statistics at several lags, windowed energy and energy profiles,
  48 DFT bins of ch0+ch1, Teager energy) -> clip + standardise -> one-hidden-layer MLP (one weight set per
  sample_rate; group as a one-hot input) -> argmax.
  The MLP was distilled from an ensemble of the 1-D CNNs of the previous versions.
  Metadata: sample_rate and group only; level is never used.

Fallback: if numba is missing, the kernel does not compile, or it does not reproduce the reference logits
stored in hf_weights.npz at setup, the previous submission (v5: 1-D CNN, fallback_v5/) answers instead.
predict never raises: any failure returns -1 (abstain).
"""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_GROUPS = {"A": 0, "B": 1, "C": 2}

try:
    import numba as nb
    OPTS = dict(cache=False, nogil=True, fastmath={"reassoc", "contract"})
    T = 256
    NSPEC = 48
    LAGS_A = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], np.int64)
    LAGS_B = np.array([1, 2, 4, 8, 16], np.int64)
    NF = 228
    VERSION = 4
    F0 = np.float32(0.0)
    F1 = np.float32(1.0)


    def tables():
        """CW/SW[d, j, t]: Hann-windowed cos/sin at j/256 cycles per (base-rate) sample; d = 0 (L=256), 1 (L=128)."""
        CW = np.zeros((2, NSPEC, T), np.float32)
        SW = np.zeros((2, NSPEC, T), np.float32)
        j = np.arange(1, NSPEC + 1)[:, None]
        for d, L in ((0, 256), (1, 128)):
            t = np.arange(L)[None, :]
            h = np.hanning(L)[None, :] / np.sqrt(L)
            CW[d, :, :L] = np.cos(2 * np.pi * j * t / 256) * h
            SW[d, :, :L] = np.sin(2 * np.pi * j * t / 256) * h
        return CW, SW


    @nb.njit(**OPTS)
    def _fill(x, mask):
        C, Tn = x.shape
        for r in range(C):
            j = 0
            while j < Tn:
                if not mask[r, j]:
                    j += 1
                    continue
                s = j
                while j < Tn and mask[r, j]:
                    j += 1
                e = j - 1
                p = s - 1
                q = e + 1
                if p < 0 and q >= Tn:
                    for c in range(s, e + 1):
                        x[r, c] = F0
                elif p < 0:
                    v = x[r, q]
                    for c in range(s, e + 1):
                        x[r, c] = v
                elif q >= Tn:
                    v = x[r, p]
                    for c in range(s, e + 1):
                        x[r, c] = v
                else:
                    xp = x[r, p]
                    dx = x[r, q] - xp
                    d = q - p
                    for c in range(s, e + 1):
                        x[r, c] = xp + dx * np.float32((c - p) / d)


    @nb.njit(**OPTS)
    def _xc(u, a, b, lag, L):
        """sum_t u[a, t] * u[b, t + lag] / L"""
        s = F0
        if lag >= 0:
            for t in range(L - lag):
                s += u[a, t] * u[b, t + lag]
        else:
            for t in range(L + lag):
                s += u[a, t - lag] * u[b, t]
        return s / np.float32(L)


    @nb.njit(**OPTS)
    def _std_inplace(v, L):
        m = F0
        for t in range(L):
            m += v[t]
        m /= np.float32(L)
        s = F0
        for t in range(L):
            d = v[t] - m
            s += d * d
        s = np.float32(np.sqrt(s / np.float32(L)))
        if not (s > np.float32(1e-6)):
            s = np.float32(1e-6)
        inv = F1 / s
        for t in range(L):
            v[t] = (v[t] - m) * inv


    @nb.njit(**OPTS)
    def feats(src, dec, grp, k, x, mask, sbuf, u, CW, SW, f):
        C = 4
        lrs = np.empty(4, np.float32)
        # ---- 1. copy, mask non-finite, fill
        bad = False
        for r in range(C):
            for c in range(T):
                v = np.float32(src[r, c])
                x[r, c] = v
                b = not np.isfinite(v)
                mask[r, c] = b
                bad = bad or b
        if bad:
            _fill(x, mask)
        # ---- 2. scale by winsorised mean / std (no sort, branch-free passes), despike
        kf = np.float32(k)
        nbad = 0
        for r in range(C):
            xr = x[r]
            m = F0
            for c in range(T):
                m += xr[c]
            m *= np.float32(1.0 / T)
            s = F0
            for c in range(T):
                d = xr[c] - m
                s += d * d
            s = np.float32(np.sqrt(s * np.float32(1.0 / T)))
            for it in range(2):
                lo = m - np.float32(3.5) * s
                hi = m + np.float32(3.5) * s
                sm = F0
                for c in range(T):
                    v = xr[c]
                    v = lo if v < lo else v
                    v = hi if v > hi else v
                    sm += v
                m2 = sm * np.float32(1.0 / T)
                ss = F0
                for c in range(T):
                    v = xr[c]
                    v = lo if v < lo else v
                    v = hi if v > hi else v
                    e = v - m2
                    ss += e * e
                m = m2
                s = np.float32(np.sqrt(ss * np.float32(1.0 / T)))
            if not (s >= np.float32(1e-20)):
                s = np.float32(1e-20)
            lrs[r] = np.float32(np.log(s))
            inv = F1 / s
            for c in range(T):
                xr[c] = (xr[c] - m) * inv
            for c in range(T):
                nbad += 0 if abs(xr[c]) <= kf else 1
        if nbad > 0:
            for r in range(C):
                for c in range(T):
                    mask[r, c] = not (abs(x[r, c]) <= kf)
            _fill(x, mask)
            for r in range(C):
                for c in range(T):
                    v = x[r, c]
                    if v < -kf:
                        x[r, c] = -kf
                    elif v > kf:
                        x[r, c] = kf
        # ---- 3. decimate, 4. standardise
        if dec:
            L = 128
            for r in range(C):
                for i in range(L):
                    u[r, i] = (x[r, 2 * i] + x[r, 2 * i + 1]) * np.float32(0.5)
        else:
            L = 256
            for r in range(C):
                for i in range(L):
                    u[r, i] = x[r, i]
        for r in range(C):
            _std_inplace(u[r], L)
        Lf = np.float32(L)
        n = 0
        # ---- A. relative log scale of the channels (4)
        ml = (lrs[0] + lrs[1] + lrs[2] + lrs[3]) * np.float32(0.25)
        for r in range(C):
            f[n] = lrs[r] - ml
            n += 1
        # ---- C. cross-correlation ch0-ch1 at lags -16..16 (33) + max, argmax (2)
        best = np.float32(-2.0)
        bl = 0
        for lag in range(-16, 17):
            v = _xc(u, 0, 1, lag, L)
            f[n] = v
            n += 1
            if v > best:
                best = v
                bl = lag
        f[n] = best
        f[n + 1] = np.float32(bl / 16.0)
        n += 2
        # ---- D. autocorrelations (22 + 10)
        for r in range(2):
            for i in range(LAGS_A.shape[0]):
                f[n] = _xc(u, r, r, LAGS_A[i], L)
                n += 1
        for r in range(2, 4):
            for i in range(LAGS_B.shape[0]):
                f[n] = _xc(u, r, r, LAGS_B[i], L)
                n += 1
        # ---- F. increments: dstd, dskew, dkurt, positive-variance fraction, skew, kurt, ddstd (7 x 4)
        for r in range(C):
            s2 = F0; s3 = F0; s4 = F0; sp = F0; u3 = F0; u4 = F0; dd2 = F0
            for t in range(L - 1):
                d = u[r, t + 1] - u[r, t]
                d2 = d * d
                s2 += d2
                s3 += d2 * d
                s4 += d2 * d2
                if d > F0:
                    sp += d2
            for t in range(L):
                v = u[r, t]
                v2 = v * v
                u3 += v2 * v
                u4 += v2 * v2
            for t in range(1, L - 1):
                d = u[r, t + 1] - np.float32(2.0) * u[r, t] + u[r, t - 1]
                dd2 += d * d
            m2 = s2 / np.float32(L - 1) + np.float32(1e-9)
            ds = np.float32(np.sqrt(m2))
            f[n] = ds
            f[n + 1] = (s3 / np.float32(L - 1)) / (m2 * ds)
            f[n + 2] = np.float32(np.log((s4 / np.float32(L - 1)) / (m2 * m2) + np.float32(1e-6)))
            f[n + 3] = sp / (s2 + np.float32(1e-9))
            f[n + 4] = u3 / Lf
            f[n + 5] = np.float32(np.log(u4 / Lf + np.float32(1e-6)))
            f[n + 6] = np.float32(np.sqrt(dd2 / np.float32(L - 2)))
            n += 7
        # ---- F2. increments at lags 2, 4, 8: std, skew, log-kurtosis, 4 channels (36)
        for r in range(C):
            ur = u[r]
            for lag in (2, 4, 8):
                a2 = F0; a3 = F0; a4 = F0
                for t in range(L - lag):
                    d = ur[t + lag] - ur[t]
                    d2 = d * d
                    a2 += d2
                    a3 += d2 * d
                    a4 += d2 * d2
                cnt = np.float32(L - lag)
                m2 = a2 / cnt + np.float32(1e-9)
                sd = np.float32(np.sqrt(m2))
                f[n] = sd
                f[n + 1] = (a3 / cnt) / (m2 * sd)
                f[n + 2] = np.float32(np.log((a4 / cnt) / (m2 * m2) + np.float32(1e-6)))
                n += 3
        # ---- s01 = standardised ch0 + ch1 in w
        w = u[4]
        for t in range(L):
            w[t] = u[0, t] + u[1, t]
        _std_inplace(w, L)
        # ---- H. windowed energy (windows of 16) and autocorrelation of the squares: ch0, ch1, s01 (3 x 6)
        nw = L // 16
        for sig in (0, 1, 4):
            us = u[sig]
            sm = F0; sq = F0; mn = np.float32(1e30); mx = F0; sl = F0; sl2 = F0
            for i in range(nw):
                e = F0
                for t in range(16 * i, 16 * i + 16):
                    e += us[t] * us[t]
                sd = np.float32(np.sqrt(e * np.float32(1.0 / 16.0)))
                sm += sd
                sq += sd * sd
                if sd < mn:
                    mn = sd
                if sd > mx:
                    mx = sd
                lg = np.float32(np.log(sd + np.float32(1e-3)))
                sl += lg
                sl2 += lg * lg
            mu = sm / np.float32(nw)
            var = sq / np.float32(nw) - mu * mu
            if var < F0:
                var = F0
            f[n] = np.float32(np.sqrt(var)) / (mu + np.float32(1e-6))
            f[n + 1] = mn / (mx + np.float32(1e-6))
            lv = sl2 / np.float32(nw) - (sl / np.float32(nw)) ** 2
            if lv < F0:
                lv = F0
            f[n + 2] = np.float32(np.sqrt(lv))
            n += 3
            den = F0
            for t in range(L):
                q = us[t] * us[t] - F1
                sbuf[t] = q
                den += q * q
            den += np.float32(1e-6)
            for lag in (1, 4, 16):
                s = F0
                for t in range(L - lag):
                    s += sbuf[t] * sbuf[t + lag]
                f[n] = s / den
                n += 1
        # ---- I. spectrum of s01: 48 log relative powers + entropy, peak fraction, peak position, centroid (52)
        d = 1 if dec else 0
        tot = np.float32(1e-9)
        pmx = F0
        jm = 0
        for j in range(NSPEC):
            re = F0; im = F0
            for t in range(L):
                re += CW[d, j, t] * w[t]
                im += SW[d, j, t] * w[t]
            p = re * re + im * im
            f[n + j] = p
            tot += p
            if p > pmx:
                pmx = p
                jm = j
        ent = F0
        cen = F0
        for j in range(NSPEC):
            p = f[n + j] / tot
            ent -= p * np.float32(np.log(p + np.float32(1e-9)))
            cen += p * np.float32(j)
            f[n + j] = np.float32(np.log(p + np.float32(1e-4)))
        n += NSPEC
        f[n] = ent
        f[n + 1] = pmx / tot
        f[n + 2] = np.float32(jm / 48.0)
        f[n + 3] = cen / np.float32(48.0)
        n += 4
        # ---- J. Teager energy psi = v[t]^2 - v[t-1] v[t+1]: mean and CV for ch0, ch1, s01 (6)
        for sig in (0, 1, 4):
            us = u[sig]
            s1 = F0; s2 = F0
            for t in range(1, L - 1):
                ps = us[t] * us[t] - us[t - 1] * us[t + 1]
                s1 += ps
                s2 += ps * ps
            mu = s1 / np.float32(L - 2)
            var = s2 / np.float32(L - 2) - mu * mu
            if var < F0:
                var = F0
            f[n] = mu
            f[n + 1] = np.float32(np.sqrt(var)) / (abs(mu) + np.float32(1e-3))
            n += 2
        # ---- P. smoothed energy profile of s01 (windows of 32, hop 8): CV, min/max, acf at 6 lags (8)
        ne = (L - 32) // 8 + 1
        mu = F0
        for i in range(ne):
            e = F0
            for t in range(8 * i, 8 * i + 32):
                e += w[t] * w[t]
            sbuf[i] = e
            mu += e
        mu /= np.float32(ne)
        var = F0; mn = np.float32(1e30); mx = F0
        for i in range(ne):
            e = sbuf[i]
            if e < mn:
                mn = e
            if e > mx:
                mx = e
            sbuf[i] = e - mu
            var += sbuf[i] * sbuf[i]
        f[n] = np.float32(np.sqrt(var / np.float32(ne))) / (mu + np.float32(1e-6))
        f[n + 1] = mn / (mx + np.float32(1e-6))
        n += 2
        for lag in (1, 2, 3, 4, 6, 8):
            s = F0
            for i in range(ne - lag):
                s += sbuf[i] * sbuf[i + lag]
            f[n] = s / (var + np.float32(1e-6))
            n += 1
        # ---- P2. finer energy profile of s01 (windows of 16, hop 4): CV, min/max, acf at 4 lags (6)
        ne = (L - 16) // 4 + 1
        mu = F0
        for i in range(ne):
            e = F0
            for t in range(4 * i, 4 * i + 16):
                e += w[t] * w[t]
            sbuf[i] = e
            mu += e
        mu /= np.float32(ne)
        var = F0; mn = np.float32(1e30); mx = F0
        for i in range(ne):
            e = sbuf[i]
            if e < mn:
                mn = e
            if e > mx:
                mx = e
            sbuf[i] = e - mu
            var += sbuf[i] * sbuf[i]
        f[n] = np.float32(np.sqrt(var / np.float32(ne))) / (mu + np.float32(1e-6))
        f[n + 1] = mn / (mx + np.float32(1e-6))
        n += 2
        for lag in (1, 2, 4, 8):
            s = F0
            for i in range(ne - lag):
                s += sbuf[i] * sbuf[i + lag]
            f[n] = s / (var + np.float32(1e-6))
            n += 1
        # ---- K. group one-hot (3)
        f[n] = F1 if grp == 0 else F0
        f[n + 1] = F1 if grp == 1 else F0
        f[n + 2] = F1 if grp == 2 else F0
        n += 3
        return n


    @nb.njit(**OPTS)
    def predict_k(src, dec, grp, k, x, mask, sbuf, u, CW, SW, f, lo, hi, mu, isd, W0, b0, W1, b1, h, o):
        """Features -> clip, standardise -> MLP (one hidden ReLU layer) -> argmax. Per-sr parameter sets (index 0/1).
        Leaves the 8 logits in o. Returns -1 if a logit is not finite."""
        feats(src, dec, grp, k, x, mask, sbuf, u, CW, SW, f)
        d = 1 if dec else 0
        nf = f.shape[0]
        for i in range(nf):
            v = f[i]
            v = lo[d, i] if not (v >= lo[d, i]) else v
            v = hi[d, i] if v > hi[d, i] else v
            f[i] = (v - mu[d, i]) * isd[d, i]
        H = b0.shape[1]
        for j in range(H):
            a = b0[d, j]
            for i in range(nf):
                a += W0[d, j, i] * f[i]
            h[j] = a if a > F0 else F0
        best = 0
        bv = np.float32(-np.inf)
        ok = True
        for c in range(8):
            a = b1[d, c]
            for j in range(H):
                a += W1[d, c, j] * h[j]
            o[c] = a
            if not np.isfinite(a):
                ok = False
            if a > bv:
                bv = a
                best = c
        return best if ok else -1


    def make_kernel(nf, hid):
        """predict_p(src, dec, grp, P, wk, mask): the whole model in one call with 6 arguments (3 arrays).
        P packs every read-only parameter, wk every float32 work buffer (the 8 logits are its last 8 slots)."""
        sz = [2 * NSPEC * T, 2 * NSPEC * T, 2 * nf, 2 * nf, 2 * nf, 2 * nf, 2 * hid * nf, 2 * hid, 2 * 8 * hid, 2 * 8]
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [int(v) for v in np.concatenate([[0], np.cumsum(sz)])]
        w1 = 4 * T; w2 = w1 + 5 * T; w3 = w2 + T; w4 = w3 + nf; w5 = w4 + hid; w6 = w5 + 8
        kd = 8.0

        @nb.njit(**OPTS)
        def predict_p(src, dec, grp, P, wk, mask):
            if src.shape[0] != 4 or src.shape[1] != T:
                return -1
            return predict_k(src, dec, grp, kd, wk[0:w1].reshape(4, T), mask, wk[w2:w3], wk[w1:w2].reshape(5, T),
                             P[p0:p1].reshape(2, NSPEC, T), P[p1:p2].reshape(2, NSPEC, T), wk[w3:w4],
                             P[p2:p3].reshape(2, nf), P[p3:p4].reshape(2, nf), P[p4:p5].reshape(2, nf),
                             P[p5:p6].reshape(2, nf), P[p6:p7].reshape(2, hid, nf), P[p7:p8].reshape(2, hid),
                             P[p8:p9].reshape(2, 8, hid), P[p9:p10].reshape(2, 8), wk[w4:w5], wk[w5:w6])
        return predict_p, p10, w6
    _HAVE = True
except Exception:  # numba missing or broken -> fallback model
    _HAVE = False


def _load_fallback():
    spec = importlib.util.spec_from_file_location("gdg_fallback_v5", str(HERE / "fallback_v5" / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model()


class Model:
    def __init__(self):
        self._fast = False
        self._fb = None
        try:
            if _HAVE:
                self._setup_fast()
        except Exception:
            self._fast = False
        if not self._fast:
            self._fb = _load_fallback()
        try:
            gc.collect()
            gc.freeze()
        except Exception:
            pass

    def _setup_fast(self):
        w = np.load(HERE / "hf_weights.npz")
        nf = int(w["mu"].shape[1])
        hid = int(w["b0"].shape[1])
        if nf != NF:
            return
        CW, SW = tables()
        k, psize, wsize = make_kernel(nf, hid)
        P = np.concatenate([np.asarray(a, np.float32).ravel() for a in
                            (CW, SW, w["lo"], w["hi"], w["mu"], w["isd"], w["W0"], w["b0"], w["W1"], w["b1"])])
        if P.shape[0] != psize:
            return
        self._P = np.ascontiguousarray(P)
        self._wk = np.zeros(wsize, np.float32)
        self._mask = np.zeros((4, 256), np.bool_)
        self._o = self._wk[wsize - 8:]
        # Sample.series is always a read-only C-contiguous float32 array: the kernel is compiled for exactly
        # that type. Anything else goes through _slow, which copies it into this buffer (same type).
        self._x = np.empty((4, 256), np.float32)
        self._xro = self._x.view()
        self._xro.setflags(write=False)
        self._k = k
        # compile (untimed setup) and check the kernel against the reference logits computed at export
        tx, tdec, tg = w["test_x"], w["test_dec"], w["test_g"]
        ref = w["ref_logits"] if "ref_logits" in w.files else None
        for i in range(len(tx)):
            np.copyto(self._x, tx[i], casting="unsafe")
            c = self._k(self._xro, bool(tdec[i]), int(tg[i]), self._P, self._wk, self._mask)
            if ref is not None:
                r = ref[i]
                if c != int(r.argmax()) or not np.allclose(self._o, r, rtol=0.0, atol=2e-2):
                    return
        try:  # only the signature compiled above is ever used; anything else must fail fast, not compile
            self._k.disable_compile()
        except Exception:
            pass
        self._fast = True

    def _slow(self, x) -> int:
        """Unusual inputs (other array type / layout, missing or odd metadata): same kernel through our buffer."""
        try:
            np.copyto(self._x, x.series, casting="unsafe")         # raises for any shape other than (4, 256)
            m = x.meta
            try:
                dec = int(round(float(m.get("sample_rate", 1)))) == 2
            except Exception:
                dec = False
            g = _GROUPS.get(str(m.get("group", "B")).strip().upper(), 1)
            return int(self._k(self._xro, dec, g, self._P, self._wk, self._mask))
        except Exception:
            return -1

    def predict(self, x) -> int:
        if self._fast:
            try:
                m = x.meta
                return self._k(x.series, m["sample_rate"] == 2, _GROUPS[m["group"]], self._P, self._wk, self._mask)
            except Exception:
                return self._slow(x)
        try:
            return int(self._fb.predict(x))
        except Exception:
            return -1

    def logits(self, x) -> np.ndarray:
        """Class scores behind predict (evaluation only; the harness never calls it)."""
        if self._fast:
            self.predict(x)
            return self._o.copy()
        return self._fb.logits(x)
