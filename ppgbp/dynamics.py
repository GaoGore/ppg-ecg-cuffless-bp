"""Beat-to-beat dynamics within a 1-minute segment.

Per-beat series: RR interval, pulse arrival time (PAT, max-upslope point after
the R peak) of the deep (IR) and superficial (green) channel groups, and the
IR pulse amplitude. From them:
* variability spectra (LF 0.04-0.15 Hz, HF 0.15-0.4 Hz) of RR, PAT and amplitude,
  and the dominant breathing frequency;
* baroreflex-like coupling: a fall in PAT means a rise in pressure, so
  -PAT is the pressure proxy; lagged beat correlations of -PAT (and amplitude)
  with later RR intervals, cross-spectral coherence and transfer gain in LF/HF,
  and the spontaneous-sequence slope (runs of >=3 beats where -PAT and the next
  RR move together);
* the beat-level dependence of PAT on RR (pre-ejection period changes).
Segments with an unusable ECG (no R-wave timing) return NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import coherence, csd, savgol_filter, welch

from . import beats as bt

FS = bt.FS
W0 = bt.W0
GRID_FS = 4.0
NAN = float("nan")
LF, HF = (0.04, 0.15), (0.15, 0.40)


def _group_beats(seg, chans):
    """Per-beat mean of amplitude-normalised windows of the good channels in `chans`."""
    q = seg["ch_quality"]
    use = [c for c in chans if q[c] >= 0.3]
    if not use:
        return None
    B = seg["beats_ppg"][:, use, :]
    T = seg["templates"][use]
    amp = np.nanmax(T[:, W0:], axis=1) - np.nanmin(T[:, W0:W0 + int(0.5 * FS)], axis=1)
    amp = np.where(amp > 0, amp, np.nan)
    return np.nanmean(B / amp[None, :, None], axis=1)                 # (n_beats, WIN)


def _beat_pat_amp(X, med_rr):
    """Max-upslope time (s after R, sub-sample) and pulse amplitude per beat."""
    a, b = W0 + int(0.03 * FS), W0 + int(min(0.5, 0.9 * med_rr) * FS)
    d1 = savgol_filter(X, 15, 3, deriv=1, axis=1)[:, a:b]
    i = np.argmax(d1, axis=1)
    off = np.zeros(len(i))
    ok = (i > 0) & (i < d1.shape[1] - 1)
    r = np.arange(len(i))
    y0, y1, y2 = d1[r[ok], i[ok] - 1], d1[r[ok], i[ok]], d1[r[ok], i[ok] + 1]
    den = y0 - 2 * y1 + y2
    off[ok] = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1), 0)
    pat = (a - W0 + i + off) / FS
    j = a + i
    lo = np.array([X[k, max(0, jj - int(0.3 * FS)):jj + 1].min() for k, jj in enumerate(j)])
    hi = np.array([X[k, jj:min(X.shape[1], jj + int(0.35 * FS))].max() for k, jj in enumerate(j)])
    return pat, hi - lo


def _resample(t, v, valid):
    """Evenly resampled (4 Hz), linearly detrended series from valid beats."""
    tv, vv = t[valid], v[valid]
    if len(tv) < 10:
        return None
    g = np.arange(tv[0], tv[-1], 1 / GRID_FS)
    x = np.interp(g, tv, vv)
    return x - np.polyval(np.polyfit(g, x, 1), g)


def _band(f, p, band):
    m = (f >= band[0]) & (f < band[1])
    return float(np.trapezoid(p[m], f[m])) if m.any() else NAN


def _spec(x, name, out, scale=1.0):
    f, p = welch(x * scale, fs=GRID_FS, nperseg=min(len(x), 128))
    lf, hf = _band(f, p, LF), _band(f, p, HF)
    out[f"{name}_lf"], out[f"{name}_hf"] = lf, hf
    out[f"{name}_lfnu"] = lf / (lf + hf) if lf + hf > 0 else NAN
    return f, p


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return NAN
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _sequence_brs(p, rr, valid):
    """Spontaneous-sequence method on (-PAT in ms, next RR in ms)."""
    x, y = -p[:-1] * 1000, rr[1:] * 1000
    ok = valid[:-1] & valid[1:]
    dx, dy = np.diff(x), np.diff(y)
    same = (np.sign(dx) == np.sign(dy)) & (np.abs(dx) > 0.5) & (np.abs(dy) > 1.0) & ok[1:] & ok[:-1]
    slopes, i = [], 0
    while i < len(same):
        if same[i]:
            j = i
            while j + 1 < len(same) and same[j + 1] and np.sign(dx[j + 1]) == np.sign(dx[i]):
                j += 1
            if j - i + 1 >= 2:                                   # >= 3 beats
                xs, ys = x[i:j + 2], y[i:j + 2]
                if np.std(xs) > 0 and abs(np.corrcoef(xs, ys)[0, 1]) > 0.8:
                    slopes.append(np.polyfit(xs, ys, 1)[0])
            i = j + 1
        else:
            i += 1
    return (float(np.median(slopes)) if slopes else NAN), len(slopes)


def dynamics_features(seg: dict) -> dict:
    keys = ["valid_frac", "pat_ir_sd_ms", "pat_g_sd_ms", "amp_cv", "resp_f", "pat_rr_slope",
            "brs_seq", "n_seq", "coh_lf", "coh_hf", "gain_lf", "gain_hf"]
    keys += [f"{s}_{b}" for s in ("rr", "patir", "patg", "amp") for b in ("lf", "hf", "lfnu")]
    keys += [f"xc{k}" for k in range(4)] + [f"xa{k}" for k in range(4)]
    out = dict.fromkeys(keys, NAN)
    if not seg.get("ok") or not seg.get("ecg_ok"):
        return out
    r = seg["r"]
    t = r / FS
    rr = np.r_[np.nan, np.diff(t)]
    med_rr = seg["med_rr"]
    Xir = _group_beats(seg, range(0, 4))
    Xg = _group_beats(seg, range(8, 12))
    if Xir is None:
        return out
    p_ir, amp = _beat_pat_amp(Xir, med_rr)
    p_g = _beat_pat_amp(Xg, med_rr)[0] if Xg is not None else np.full(len(t), np.nan)
    valid = seg["regular"] & seg["still"] & np.isfinite(rr)
    for p in (p_ir,):
        valid &= np.abs(p - np.nanmedian(p[valid])) < 0.06
    out["valid_frac"] = float(valid.mean())
    if valid.sum() < 15:
        return out
    out["pat_ir_sd_ms"] = float(np.std(p_ir[valid]) * 1000)
    vg = valid & np.isfinite(p_g) & (np.abs(p_g - np.nanmedian(p_g[valid])) < 0.06)
    out["pat_g_sd_ms"] = float(np.std(p_g[vg]) * 1000) if vg.sum() > 10 else NAN
    amp_n = amp / np.nanmedian(amp[valid])
    out["amp_cv"] = float(np.std(amp_n[valid]))
    out["pat_rr_slope"] = float(np.polyfit(rr[valid], p_ir[valid], 1)[0]) if np.std(rr[valid]) > 0 else NAN
    # spectra on 4 Hz resampled, detrended series (RR, PAT in ms; amplitude in relative units)
    s_rr, s_ir = _resample(t, rr * 1000, valid), _resample(t, p_ir * 1000, valid)
    s_g = _resample(t, p_g * 1000, vg) if vg.sum() > 10 else None
    s_amp = _resample(t, amp_n, valid)
    if s_rr is None or s_ir is None:
        return out
    f, prr = _spec(s_rr, "rr", out)
    _spec(s_ir, "patir", out)
    if s_g is not None:
        _spec(s_g, "patg", out)
    fa, pa = _spec(s_amp, "amp", out)
    resp = (f >= 0.1) & (f <= 0.5)
    out["resp_f"] = float(f[resp][np.argmax(prr[resp] + np.interp(f[resp], fa, pa / (pa.max() + 1e-12)) * prr.max())])
    # coupling: pressure proxy (-PAT) -> RR
    n = min(len(s_rr), len(s_ir))
    x, y = -s_ir[:n], s_rr[:n]
    nper = min(n, 128)
    fc, coh = coherence(x, y, fs=GRID_FS, nperseg=nper)
    _, pxy = csd(x, y, fs=GRID_FS, nperseg=nper)
    _, pxx = welch(x, fs=GRID_FS, nperseg=nper)
    gain = np.abs(pxy) / np.where(pxx > 0, pxx, np.nan)
    for band, nm in ((LF, "lf"), (HF, "hf")):
        m = (fc >= band[0]) & (fc < band[1])
        if m.any():
            out[f"coh_{nm}"] = float(np.mean(coh[m]))
            w = coh[m]
            out[f"gain_{nm}"] = float(np.nansum(gain[m] * w) / w.sum()) if w.sum() > 0 else NAN
    # lagged beat correlations
    pv = np.where(valid, -p_ir, np.nan)
    av = np.where(valid, amp_n, np.nan)
    rv = np.where(valid, rr, np.nan)
    for k in range(4):
        out[f"xc{k}"] = _corr(pv[:len(pv) - k], rv[k:])
        out[f"xa{k}"] = _corr(av[:len(av) - k], rv[k:])
    out["brs_seq"], out["n_seq"] = _sequence_brs(p_ir, rr, valid)
    return out
