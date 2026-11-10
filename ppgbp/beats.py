"""Beat-level processing of one 1-minute segment.

Pipeline: ECG R-peaks (energy envelope + rank-based threshold + polarity-aware
refinement) -> R-aligned beat windows for all 16 PPG channels and ECG ->
per-beat quality gating (ECG rhythm, accelerometer motion, template
correlation) -> robust ensemble templates at 500 Hz.

PPG raw values are detected light intensity, which falls when blood volume
rises, so the pulse waveform is the negated, band-passed signal.
If the ECG is unusable, beats are aligned on the PPG upstroke instead and
`ecg_ok` is False (timing relative to the R wave is then meaningless).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

FS = 500
PRE, POST = 0.20, 1.40                       # beat window around R (s)
W0, W1 = int(PRE * FS), int(POST * FS)       # 100, 700 samples
WIN = W0 + W1                                # 800 samples
T_AXIS = (np.arange(WIN) - W0) / FS          # seconds relative to R


@lru_cache(maxsize=None)
def _sos(lo, hi, fs=FS, order=3):
    return butter(order, (lo, hi), btype="bandpass", fs=fs, output="sos")


def _parabolic(y, i):
    """Sub-sample offset of an extremum at integer index i."""
    if i <= 0 or i >= len(y) - 1:
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return 0.0 if den == 0 else 0.5 * (a - c) / den


def ppg_pulses(ppg_raw: np.ndarray, lo=0.3, hi=15.0) -> tuple[np.ndarray, np.ndarray]:
    """(16, L) raw intensities -> (pulse waves, DC levels). Pulse = -bandpassed."""
    dc = np.median(ppg_raw, axis=1)
    x = ppg_raw - dc[:, None]
    good = np.isfinite(x).all(axis=1)
    out = np.full_like(x, np.nan)
    if good.any():
        out[good] = -sosfiltfilt(_sos(lo, hi), x[good], axis=1)
    return out, dc


def spectral_hr(x: np.ndarray, fs=FS) -> float:
    """Pulse rate by harmonic summation (fundamental + 2 harmonics) of the PSD."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), 8192))
    band = (f >= 0.6) & (f <= 3.3)
    fb = f[band]
    score = p[band] + np.interp(2 * fb, f, p) + 0.5 * np.interp(3 * fb, f, p)
    return float(60 * fb[np.argmax(score)])


def detect_r_peaks(ecg: np.ndarray, hr_hint: float | None = None, fs=FS) -> dict:
    """R-peak fiducials (float sample indices) plus quality descriptors."""
    res = {"r": np.array([]), "polarity": np.nan, "qrs_corr": np.nan, "regular_frac": np.nan,
           "count_ratio": np.nan, "ok": False}
    if not np.isfinite(ecg).all() or np.std(ecg) == 0:
        return res
    b = sosfiltfilt(_sos(5.0, 30.0), ecg - np.median(ecg))
    d = np.gradient(b)
    k = int(0.08 * fs)
    env = np.convolve(d * d, np.ones(k) / k, mode="same")
    cand, _ = find_peaks(env, distance=int(0.25 * fs))
    if len(cand) < 10:
        return res
    # heart rate from the periodicity of the QRS energy itself; the PPG hint is
    # only a fallback (it can lock onto sub-harmonics under motion)
    hr_env, strength = _envelope_hr(env, fs)
    hr_use = hr_env if strength > 0.25 else (hr_hint if hr_hint and np.isfinite(hr_hint) else 75.0)
    res["hr_env"], res["env_strength"] = hr_env, strength
    n_exp = int(round(hr_use * len(ecg) / fs / 60))
    n_exp = int(np.clip(n_exp, 20, len(cand)))
    ref = np.sort(env[cand])[::-1][min(n_exp, len(cand)) - 1]
    sel = cand[env[cand] >= 0.4 * ref]
    # refractory clean-up: within 0.6 * typical RR keep the higher-energy candidate
    rr_typ = 60.0 / hr_use * fs
    hr_hint = hr_use
    keep = []
    for p in sel:
        if keep and p - keep[-1] < 0.6 * rr_typ:
            if env[p] > env[keep[-1]]:
                keep[-1] = p
        else:
            keep.append(p)
    sel = np.array(keep)
    if len(sel) < 10:
        return res
    # polarity: compare signed extremes of the QRS band signal around detections
    h = int(0.06 * fs)
    seg = np.array([b[max(0, p - h):p + h] for p in sel if p - h >= 0 and p + h <= len(b)])
    if len(seg) < 10:
        return res
    pol = 1.0 if np.median(seg.max(axis=1)) >= np.median(-seg.min(axis=1)) else -1.0
    r = []
    for p in sel:
        lo, hi = max(0, p - h), min(len(b), p + h)
        i = lo + int(np.argmax(pol * b[lo:hi]))
        r.append(i + _parabolic(pol * b, i))
    r = np.unique(np.array(r))
    # QRS template correlation, drop artefacts
    ri = np.round(r).astype(int)
    m = (ri - h >= 0) & (ri + h < len(b))
    r, ri = r[m], ri[m]
    qrs = np.stack([b[i - h:i + h] for i in ri])
    tmpl = np.median(qrs, axis=0)
    qc = _rowcorr(qrs, tmpl)
    good = qc > 0.6
    r, qc = r[good], qc[good]
    if len(r) < 10:
        return res
    rr = np.diff(r) / fs
    med = np.median(rr)
    reg = (np.abs(rr / med - 1) < 0.2)
    res.update(r=r, polarity=pol, qrs_corr=float(np.median(qc)), regular_frac=float(reg.mean()),
               count_ratio=float(len(r) / max(1.0, len(ecg) / fs * (hr_hint or 60 / med) / 60)))
    res["ok"] = bool(res["regular_frac"] > 0.6 and 0.75 < res["count_ratio"] < 1.33 and 0.3 < med < 1.6)
    return res


def _envelope_hr(env: np.ndarray, fs=FS) -> tuple[float, float]:
    """Dominant beat period of the QRS energy envelope via autocorrelation
    (50 Hz decimated, artefacts clipped). Returns (bpm, normalised peak)."""
    e = np.minimum(env, np.percentile(env, 99))[::10]
    e = e - e.mean()
    n = len(e)
    F = np.fft.rfft(e, 2 * n)
    ac = np.fft.irfft(F * np.conj(F))[:n]
    if ac[0] <= 0:
        return float("nan"), 0.0
    ac = ac / ac[0]
    efs = fs / 10
    lo, hi = int(0.33 * efs), int(1.6 * efs)
    i = lo + int(np.argmax(ac[lo:hi]))
    half = i // 2
    if half >= lo and ac[half] > 0.6 * ac[i]:      # period doubled by alternating beats
        i = half
    return float(60 * efs / i), float(ac[i])


def ppg_fiducials(pulse: np.ndarray, hr: float, fs=FS) -> np.ndarray:
    """Fallback beat reference: max-upslope points of a PPG pulse wave, shifted
    back by a typical pulse arrival time so templates sit where R-aligned ones do."""
    d = np.gradient(pulse)
    rr = 60.0 / hr * fs
    pk, _ = find_peaks(d, distance=int(0.5 * rr), prominence=np.std(d) * 0.5)
    return pk.astype(float) - 0.21 * fs


def _rowcorr(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=1, keepdims=True)
    t = t - t.mean()
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(t)
    return np.where(den > 0, A @ t / np.where(den > 0, den, 1), 0.0)


def beat_matrix(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Stack windows [r - PRE, r + POST) for each (float) fiducial: (n_beats, WIN).
    Uses linear interpolation so sub-sample R timing is preserved."""
    base = np.arange(-W0, W1)
    idx = r[:, None] + base[None, :]
    i0 = np.floor(idx).astype(int)
    frac = idx - i0
    return x[i0] * (1 - frac) + x[i0 + 1] * frac


def motion_per_beat(acc: np.ndarray, r: np.ndarray, fs=FS, acc_fs=100) -> np.ndarray:
    """Max dynamic acceleration magnitude inside each beat window."""
    if acc.size == 0 or not np.isfinite(acc).all():
        return np.zeros(len(r))
    dyn = acc - np.median(acc, axis=1, keepdims=True)
    mag = np.sqrt((dyn ** 2).sum(axis=0))
    out = np.empty(len(r))
    for j, p in enumerate(r):
        a = int(max(0, (p - W0) * acc_fs / fs))
        b = int(min(len(mag), (p + W1) * acc_fs / fs + 1))
        out[j] = mag[a:b].max() if b > a else 0.0
    return out


def process_segment(ppg_raw: np.ndarray, ecg: np.ndarray, acc: np.ndarray, rng=None) -> dict:
    """Return beats, templates and quality info for one segment.

    templates: (17, WIN) robust mean of accepted beats (16 PPG pulses, ECG), 500 Hz
    beats_ppg: (n_beats, 16, WIN) accepted-beat windows (for subset templates)
    """
    pulses, dc = ppg_pulses(ppg_raw)
    finite_ch = np.isfinite(pulses).all(axis=1)
    ir_mean = np.nanmean(pulses[:4], axis=0) if finite_ch[:4].any() else np.nanmean(pulses, axis=0)
    hr_ppg = spectral_hr(np.nan_to_num(ir_mean))
    rd = detect_r_peaks(ecg, hr_hint=hr_ppg)
    ecg_ok = rd["ok"]
    ecg_band = sosfiltfilt(_sos(0.5, 40.0), ecg - np.median(ecg)) if np.isfinite(ecg).all() else np.zeros_like(ecg)
    if ecg_ok:
        r = rd["r"]
    else:
        hr_fb = rd.get("hr_env", np.nan) if rd.get("env_strength", 0) > 0.4 else hr_ppg
        r = ppg_fiducials(ir_mean, hr_fb if np.isfinite(hr_fb) else hr_ppg)
    r = r[(r - W0 >= 0) & (r + W1 + 1 < pulses.shape[1])]
    out = {"ecg_ok": ecg_ok, "hr_ppg": hr_ppg, "r_info": rd, "dc": dc, "n_beats_raw": len(r)}
    if len(r) < 8:
        out["ok"] = False
        return out
    rr = np.diff(r) / FS
    med_rr = float(np.median(rr))
    # rhythm regularity of each beat (both neighbouring intervals within 20%)
    rr_prev = np.r_[med_rr, rr]
    rr_next = np.r_[rr, med_rr]
    regular = (np.abs(rr_prev / med_rr - 1) < 0.2) & (np.abs(rr_next / med_rr - 1) < 0.2)
    motion = motion_per_beat(acc, r)
    mthr = np.median(motion) * 3 + 20
    still = motion < mthr
    B = np.stack([beat_matrix(pulses[c], r) if finite_ch[c] else np.full((len(r), WIN), np.nan)
                  for c in range(16)], axis=1)                               # (n, 16, WIN)
    E = beat_matrix(ecg_band, r)                                             # (n, WIN)
    base = regular & still
    if base.sum() < 8:
        base = still if still.sum() >= 8 else np.ones(len(r), bool)
    # per-channel template correlation over one cardiac cycle
    cyc = slice(W0, W0 + int(min(0.9, 0.95 * med_rr) * FS))
    acc_ch = np.zeros((len(r), 16), bool)
    for c in range(16):
        if not finite_ch[c]:
            continue
        X = B[:, c, cyc]
        t0 = np.median(X[base], axis=0)
        cc = _rowcorr(X, t0)
        amp = X.max(axis=1) - X.min(axis=1)
        a_med = np.median(amp[base])
        ok = base & (cc > 0.85) & (amp > 0.4 * a_med) & (amp < 2.5 * a_med)
        if ok.sum() < 8:                                                     # fall back to best half
            ok = base & (cc >= np.median(cc[base]))
        acc_ch[:, c] = ok
    T = np.full((17, WIN), np.nan)
    for c in range(16):
        if acc_ch[:, c].sum() >= 3:
            X = B[acc_ch[:, c], c]
            X = X - X[:, W0:W0 + int(0.1 * FS)].mean(axis=1, keepdims=True) # align pre-upstroke level
            T[c] = _trimmed_mean(X)
    common = base & (acc_ch.sum(axis=1) >= 8)
    if common.sum() < 8:
        common = base
    T[16] = _trimmed_mean(E[common] - np.median(E[common][:, :int(0.08 * FS)], axis=1, keepdims=True))
    out.update(ok=True, r=r, rr=rr, med_rr=med_rr, regular=regular, still=still, motion=motion,
               accepted=acc_ch, common=common, templates=T, beats_ppg=B, beats_ecg=E,
               ch_quality=acc_ch.mean(axis=0))
    return out


def _trimmed_mean(X: np.ndarray, prop=0.1) -> np.ndarray:
    n = len(X)
    if n < 5:
        return X.mean(axis=0)
    k = int(n * prop)
    S = np.sort(X, axis=0)
    return S[k:n - k].mean(axis=0)


def subset_templates(seg: dict, k: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    """k templates (k, 17, WIN), each averaging a random subset of the common beats."""
    idx = np.flatnonzero(seg["common"])
    n = max(4, int(len(idx) * frac))
    out = np.empty((k, 17, WIN), np.float32)
    for j in range(k):
        s = rng.choice(idx, size=min(n, len(idx)), replace=False)
        X = seg["beats_ppg"][s]
        X = X - X[:, :, W0:W0 + int(0.1 * FS)].mean(axis=2, keepdims=True)
        out[j, :16] = X.mean(axis=0)
        e = seg["beats_ecg"][s]
        out[j, 16] = (e - np.median(e[:, :int(0.08 * FS)], axis=1, keepdims=True)).mean(axis=0)
    return out
