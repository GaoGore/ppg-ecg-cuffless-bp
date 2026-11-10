"""Physiological features from beat templates (500 Hz, R-aligned).

Per PPG channel: pulse arrival times (tangent foot, d2 foot, max upslope,
peak), rise time, widths, areas, inflection/diastolic timing, APG (second
derivative) wave ratios, harmonic structure, cycle shape samples, perfusion.
Across channels: wavelength (depth) and position delays and contrasts.
ECG: QRS, QT/QTc, T/R ratio. Rhythm: HRV. Respiratory modulation. Motion.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from . import beats as bt
from .io import PPG_CHANNELS, WAVELENGTHS

FS = bt.FS
W0 = bt.W0
NAN = float("nan")

PULSE_KEYS = ("t_tan", "t_d2", "t_ms", "t_pk", "rise", "rise_ms", "amp_log", "pi_log",
              "w25", "w50", "w75", "sys_area", "skew", "slope_up", "slope_dn", "t_dn",
              "t_infl", "dt_infl", "ri", "si", "has_dia", "t_notch",
              "apg_b", "apg_c", "apg_d", "apg_e", "apg_agi", "apg_tb", "apg_te",
              "h2", "h3", "h4", "ph2", "ph3",
              "s10", "s20", "s30", "s40", "s50", "s60", "s70", "s80", "s90",
              "quality")


def _sub(y, i):
    return i + bt._parabolic(y, i)


def pulse_features(P: np.ndarray, med_rr: float, dc: float, height_m: float, quality: float) -> dict:
    out = dict.fromkeys(PULSE_KEYS, NAN)
    out["quality"] = quality
    if not np.isfinite(P).all() or np.ptp(P) <= 0:
        return out
    n = len(P)
    d1 = savgol_filter(P, 15, 3, deriv=1, delta=1 / FS)
    d2 = savgol_filter(P, 21, 3, deriv=2, delta=1 / FS)
    ix = lambda t: int(np.clip(round(W0 + t * FS), 0, n - 1))           # noqa: E731
    tt = lambda i: (i - W0) / FS                                        # noqa: E731
    a, b = ix(0.03), ix(min(0.5, 0.9 * med_rr))
    if b - a < 5:
        return out
    ims = a + int(np.argmax(d1[a:b]))
    lo = max(0, ims - int(0.3 * FS))
    imin = lo + int(np.argmin(P[lo:ims + 1]))
    pk_hi = min(n - 1, ims + int(0.4 * FS), imin + int(0.85 * med_rr * FS))
    if pk_hi - ims < 3:
        return out
    ipk = ims + int(np.argmax(P[ims:pk_hi]))
    A = P[ipk] - P[imin]
    if A <= 0 or d1[ims] <= 0:
        return out
    t_ms = tt(_sub(d1, ims))
    t_tan = tt(ims) - (P[ims] - P[imin]) / d1[ims]
    d2lo = max(0, imin - int(0.02 * FS))
    id2 = d2lo + int(np.argmax(d2[d2lo:ims + 1])) if ims > d2lo else imin
    t_pk = tt(_sub(P, ipk))
    out.update(t_tan=t_tan, t_d2=tt(_sub(d2, id2)), t_ms=t_ms, t_pk=t_pk, rise=t_pk - t_tan,
               rise_ms=t_ms - t_tan, amp_log=np.log10(A), pi_log=np.log10(A / abs(dc)) if dc else NAN,
               slope_up=d1[ims] / A)
    # one cardiac cycle starting at the pre-upstroke minimum
    iend = min(n - 1, imin + int(round(med_rr * FS)))
    full_cycle = imin + int(round(med_rr * FS)) <= n - 1
    cyc = (P[imin:iend + 1] - P[imin]) / A
    m = len(cyc)
    # widths at fractions of amplitude (pulse above level, around the peak)
    k_pk = ipk - imin
    for lev, key in ((0.25, "w25"), (0.5, "w50"), (0.75, "w75")):
        above = cyc >= lev
        l = k_pk
        while l > 0 and above[l - 1]:
            l -= 1
        r = k_pk
        while r < m - 1 and above[r + 1]:
            r += 1
        out[key] = (r - l) / FS if (r < m - 1 or full_cycle) else NAN
    pos = np.clip(cyc, 0, None)
    tot = pos.sum()
    out["sys_area"] = pos[:k_pk + 1].sum() / tot if tot > 0 else NAN
    if m > 10:
        z = (cyc - cyc.mean()) / (cyc.std() + 1e-12)
        out["skew"] = float(np.mean(z ** 3))
    idn = ipk + int(np.argmin(d1[ipk:iend])) if iend - ipk > 3 else ipk
    out["slope_dn"] = d1[idn] / A
    out["t_dn"] = tt(idn) - t_pk
    # inflection / diastolic wave in the descending limb
    s0, s1 = ipk + int(0.05 * FS), min(iend, imin + int(0.8 * med_rr * FS))
    if s1 - s0 > 10:
        pk1, _ = find_peaks(d1[s0:s1])
        if len(pk1):
            ii = s0 + pk1[0]
            out.update(t_infl=tt(ii), dt_infl=tt(ii) - t_pk, ri=(P[ii] - P[imin]) / A)
            if np.isfinite(height_m) and tt(ii) - t_pk > 0.02:
                out["si"] = height_m / (tt(ii) - t_pk)
        pkp, _ = find_peaks(P[s0:s1])
        out["has_dia"] = float(len(pkp) > 0)
        if len(pkp):
            idia = s0 + pkp[0]
            out["t_notch"] = tt(ipk + int(np.argmin(P[ipk:idia + 1]))) - t_pk
    # APG waves: a near the d2 foot, then alternating minima / maxima
    a_i = id2
    aval = d2[a_i]
    if aval > 0:
        seg_end = min(n - 1, ipk + int(0.3 * FS))
        mins, _ = find_peaks(-d2[a_i:seg_end])
        maxs, _ = find_peaks(d2[a_i:seg_end])
        mins, maxs = mins + a_i, maxs + a_i
        seq, cur = [], a_i
        for want in ("min", "max", "min", "max"):
            pool = mins if want == "min" else maxs
            nxt = pool[pool > cur]
            if not len(nxt):
                break
            cur = nxt[0]
            seq.append(cur)
        names = ("apg_b", "apg_c", "apg_d", "apg_e")
        for nm, i in zip(names, seq):
            out[nm] = d2[i] / aval
        if len(seq) >= 1:
            out["apg_tb"] = tt(seq[0]) - tt(a_i)
        if len(seq) == 4:
            out["apg_e"] = d2[seq[3]] / aval
            out["apg_te"] = tt(seq[3]) - tt(a_i)
            out["apg_agi"] = out["apg_b"] - out["apg_c"] - out["apg_d"] - out["apg_e"]
    # harmonics of one cycle
    if full_cycle and m > 32:
        c = cyc - np.linspace(cyc[0], cyc[-1], m)
        F = np.fft.rfft(c)
        if np.abs(F[1]) > 0 and len(F) > 4:
            out.update(h2=np.abs(F[2]) / np.abs(F[1]), h3=np.abs(F[3]) / np.abs(F[1]), h4=np.abs(F[4]) / np.abs(F[1]),
                       ph2=float(np.angle(F[2] * np.conj(F[1]) ** 2)), ph3=float(np.angle(F[3] * np.conj(F[1]) ** 3)))
        for q in range(1, 10):
            out[f"s{q * 10}"] = float(cyc[int(q / 10 * (m - 1))])
    return out


def ecg_features(E: np.ndarray, polarity: float, med_rr: float) -> dict:
    keys = ("r_amp_log", "q_rel", "s_rel", "qrs_w50", "qrs_dur", "t_rel", "t_time", "qt", "qtc_b", "qtc_f", "st_rel")
    out = dict.fromkeys(keys, NAN)
    if not np.isfinite(E).all() or not np.isfinite(polarity):
        return out
    e = polarity * E
    n = len(e)
    ix = lambda t: int(np.clip(round(W0 + t * FS), 0, n - 1))           # noqa: E731
    R = e[W0]
    if R <= 0:
        return out
    out["r_amp_log"] = np.log10(R)
    out["q_rel"] = e[ix(-0.06):W0].min() / R
    out["s_rel"] = e[W0:ix(0.08)].min() / R
    above = e >= 0.5 * R
    l = r = W0
    while l > ix(-0.06) and above[l - 1]:
        l -= 1
    while r < ix(0.06) and above[r + 1]:
        r += 1
    out["qrs_w50"] = (r - l) / FS
    d1 = np.gradient(savgol_filter(e, 9, 3))
    win = slice(ix(-0.1), ix(0.12))
    thr = 0.15 * np.abs(d1[win]).max()
    act = np.flatnonzero(np.abs(d1[win]) > thr)
    if len(act):
        on, off = act[0] + ix(-0.1), act[-1] + ix(-0.1)
        out["qrs_dur"] = (off - on) / FS
        t0, t1 = ix(0.15), ix(min(0.55, 0.8 * med_rr))
        if t1 - t0 > 10:
            iT = t0 + int(np.argmax(np.abs(e[t0:t1])))
            Tamp = e[iT]
            out["t_rel"] = Tamp / R
            out["t_time"] = (iT - W0) / FS
            sgn = np.sign(Tamp) if Tamp != 0 else 1.0
            de = np.gradient(sgn * e)
            j1 = min(n - 1, iT + int(0.15 * FS))
            if j1 - iT > 3:
                js = iT + int(np.argmin(de[iT:j1]))
                if de[js] < 0:
                    tend = js - (sgn * e[js]) / de[js]
                    qt = (tend - on) / FS
                    if 0.2 < qt < 0.7:
                        out.update(qt=qt, qtc_b=qt / np.sqrt(med_rr), qtc_f=qt / med_rr ** (1 / 3))
        out["st_rel"] = e[min(n - 1, off + int(0.06 * FS))] / R
    return out


def rhythm_features(seg: dict) -> dict:
    rr = seg["rr"]
    reg = (np.abs(rr / seg["med_rr"] - 1) < 0.2)
    rrg = rr[reg] if reg.sum() > 5 else rr
    d = np.diff(rrg)
    out = {"hr": 60 / seg["med_rr"], "sdnn": rrg.std(), "rmssd": np.sqrt(np.mean(d ** 2)) if len(d) else NAN,
           "pnn50": np.mean(np.abs(d) > 0.05) if len(d) else NAN, "rr_cv": rrg.std() / rrg.mean(),
           "irregular_frac": 1 - reg.mean()}
    # respiratory modulation from the IR-mean pulse: amplitude, baseline, interval series
    B = seg["beats_ppg"]
    acc = seg["accepted"]
    chans = [c for c in range(4) if acc[:, c].sum() >= 8] or [c for c in range(16) if acc[:, c].sum() >= 8]
    if chans:
        X = np.nanmean(B[:, chans, :] / np.nanstd(B[:, chans, :], axis=(0, 2), keepdims=True), axis=1)
        cyc = slice(W0, W0 + int(min(0.9, 0.95 * seg["med_rr"]) * FS))
        amp = X[:, cyc].max(axis=1) - X[:, cyc].min(axis=1)
        basev = X[:, cyc].min(axis=1)
        good = seg["common"]
        if good.sum() > 8:
            out["riav"] = amp[good].std() / amp[good].mean()
            out["riiv"] = basev[good].std() / amp[good].mean()
            # beat-level pulse arrival (max upslope) variability
            d1 = np.gradient(X[:, W0 + int(0.03 * FS):W0 + int(0.5 * FS)], axis=1)
            pat = (np.argmax(d1, axis=1) + int(0.03 * FS)) / FS
            pg = pat[good]
            out["pat_sd"] = pg.std()
            out["pat_iqr"] = np.subtract(*np.percentile(pg, [75, 25]))
            rr_b = np.r_[seg["med_rr"], rr][good]
            out["pat_rr_corr"] = np.corrcoef(pg, rr_b)[0, 1] if pg.std() > 0 and rr_b.std() > 0 else NAN
    return out


def motion_features(acc: np.ndarray, seg: dict) -> dict:
    out = {}
    if acc.size and np.isfinite(acc).all():
        g = np.median(acc, axis=1)
        norm = np.linalg.norm(g) + 1e-9
        out.update(gx=g[0] / norm, gy=g[1] / norm, gz=g[2] / norm)
        dyn = acc - g[:, None]
        out["motion"] = float(np.sqrt((dyn ** 2).sum(axis=0)).mean())
    out["still_frac"] = float(seg["still"].mean()) if "still" in seg else NAN
    return out


def segment_features(seg: dict, demo: dict, acc: np.ndarray) -> dict:
    """Flat feature dict for one processed segment."""
    f = {"ecg_ok": float(seg["ecg_ok"]), "hr_ppg": seg["hr_ppg"], "n_beats": seg.get("n_beats_raw", 0)}
    ri = seg["r_info"]
    f.update(ecgq_corr=ri["qrs_corr"], ecgq_reg=ri["regular_frac"], ecgq_cnt=ri["count_ratio"])
    if not seg.get("ok"):
        return f
    h_m = demo["height"] / 100 if demo.get("height") else NAN
    T = seg["templates"]
    per = {}
    for c, name in enumerate(PPG_CHANNELS):
        q = float(seg["ch_quality"][c])
        per[name] = pulse_features(T[c], seg["med_rr"], seg["dc"][c], h_m, q) if q >= 0.15 else dict.fromkeys(PULSE_KEYS, NAN)
        for k, v in per[name].items():
            f[f"{name}__{k}"] = v
    # wavelength consensus (median over positions) and global consensus
    wl = {}
    for w in WAVELENGTHS:
        names = [f"PPG_{w}_{i}" for i in (1, 2, 3, 4)]
        wl[w] = {k: np.nanmedian([per[nm][k] for nm in names]) if any(np.isfinite(per[nm][k]) for nm in names) else NAN
                 for k in PULSE_KEYS}
        for k, v in wl[w].items():
            f[f"{w}__{k}"] = v
    for k in PULSE_KEYS:
        vals = [per[nm][k] for nm in PPG_CHANNELS if np.isfinite(per[nm][k])]
        f[f"all__{k}"] = float(np.median(vals)) if vals else NAN
    # depth contrasts: other wavelengths minus IR, per position then median
    for w in ("R", "G", "B"):
        for k in ("t_tan", "t_ms", "t_pk", "rise", "w50", "sys_area", "apg_b", "apg_agi", "pi_log", "ri", "s50"):
            diffs = [per[f"PPG_{w}_{i}"][k] - per[f"PPG_IR_{i}"][k] for i in (1, 2, 3, 4)]
            diffs = [d for d in diffs if np.isfinite(d)]
            f[f"x_{w}-IR__{k}"] = float(np.median(diffs)) if diffs else NAN
    # position spread and position-specific delays (vs position 1), median over wavelengths
    for k in ("t_tan", "t_ms", "t_pk", "rise", "w50", "pi_log"):
        sds = []
        for w in WAVELENGTHS:
            v = [per[f"PPG_{w}_{i}"][k] for i in (1, 2, 3, 4)]
            v = [x for x in v if np.isfinite(x)]
            if len(v) >= 3:
                sds.append(np.std(v))
        f[f"xpos_sd__{k}"] = float(np.median(sds)) if sds else NAN
        for i in (2, 3, 4):
            d = [per[f"PPG_{w}_{i}"][k] - per[f"PPG_{w}_1"][k] for w in WAVELENGTHS]
            d = [x for x in d if np.isfinite(x)]
            f[f"xpos{i}-1__{k}"] = float(np.median(d)) if d else NAN
    # sub-sample upstroke delay by cross-correlation of wavelength-mean templates
    f.update(_xcorr_delays(T, seg))
    e = ecg_features(T[16], ri["polarity"] if seg["ecg_ok"] else NAN, seg["med_rr"])
    f.update({f"ecg__{k}": v for k, v in e.items()})
    f.update({f"rhy__{k}": v for k, v in rhythm_features(seg).items()})
    f.update({f"acc__{k}": v for k, v in motion_features(acc, seg).items()})
    return f


def _xcorr_delays(T: np.ndarray, seg: dict) -> dict:
    """Delay of each wavelength's mean upstroke relative to IR (seconds, + = later)."""
    out = {}
    q = seg["ch_quality"]
    means = {}
    for wi, w in enumerate(WAVELENGTHS):
        chans = [wi * 4 + i for i in range(4) if q[wi * 4 + i] >= 0.3 and np.isfinite(T[wi * 4 + i]).all()]
        if chans:
            X = np.stack([(T[c] - T[c].min()) / (np.ptp(T[c]) + 1e-12) for c in chans]).mean(axis=0)
            means[w] = np.gradient(X)
    if "IR" not in means:
        return {f"xc_{w}-IR": NAN for w in ("R", "G", "B")}
    ref = means["IR"]
    a, b = W0 + int(0.0 * FS), W0 + int(0.6 * FS)
    for w in ("R", "G", "B"):
        if w not in means:
            out[f"xc_{w}-IR"] = NAN
            continue
        x, y = ref[a:b] - ref[a:b].mean(), means[w][a:b] - means[w][a:b].mean()
        lags = np.arange(-40, 41)
        cc = np.array([np.dot(x[max(0, -l):len(x) - max(0, l)], y[max(0, l):len(y) - max(0, -l)]) for l in lags])
        i = int(np.argmax(cc))
        out[f"xc_{w}-IR"] = (lags[i] + bt._parabolic(cc, i)) / FS
    return out


NN_FS = 250
NN_PRE, NN_POST = 0.2, 1.0
NN_LEN = int((NN_PRE + NN_POST) * NN_FS)        # 300


def nn_template(T: np.ndarray, seg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Crop to [-0.2, 1.0] s, downsample to 250 Hz, normalise each PPG channel
    to its pre-upstroke minimum / pulse amplitude and the ECG to its R amplitude.
    Returns (17, NN_LEN) float32 and a (17,) validity mask."""
    seg_T = T[:, :int((NN_PRE + NN_POST) * FS)]
    x = seg_T.reshape(17, -1, 2).mean(axis=2)
    mask = np.isfinite(x).all(axis=1) & (np.r_[seg["ch_quality"], 1.0] >= 0.15)
    out = np.zeros_like(x, dtype=np.float32)
    full = seg["templates"]
    for c in range(16):
        if not mask[c]:
            continue
        ref = full[c, :int((NN_PRE + NN_POST) * FS)]
        lo, hi = np.nanmin(ref[W0:W0 + int(0.5 * FS)]), np.nanmax(ref[W0:])
        if hi - lo <= 0:
            mask[c] = False
            continue
        out[c] = (x[c] - lo) / (hi - lo)
    if mask[16]:
        pol = seg["r_info"]["polarity"] if seg["ecg_ok"] else 1.0
        rr = pol * full[16, W0]
        if seg["ecg_ok"] and rr > 0:
            out[16] = pol * x[16] / rr
        else:
            mask[16] = False
    return out, mask
