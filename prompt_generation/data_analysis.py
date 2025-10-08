import numpy as np
from scipy.integrate import trapezoid
import pandas as pd
from scipy.signal import butter, filtfilt, welch, find_peaks, sosfiltfilt, savgol_filter, sosfiltfilt

def _to_df(arr):
    arr = np.asarray(arr, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 3, "arr must be Nx3 (X,Y,Z)."
    df = {
        "X": arr[:,0].astype(float),
        "Y": arr[:,1].astype(float),
        "Z": arr[:,2].astype(float),
    }
    svm = np.linalg.norm(arr, axis=1)
    df["SVM"] = svm.astype(float)
    return df

def _butter_lowpass(x, fs, fc=0.3, order=4):
    b, a = butter(order, fc/(0.5*fs), btype="low")
    return filtfilt(b, a, x)

def _butter_highpass(x, fs, fc=0.3, order=4):
    b, a = butter(order, fc/(0.5*fs), btype="high")
    return filtfilt(b, a, x)

def _butter_bandpass(x, fs, f1=0.7, f2=3.0, order=4):
    b, a = butter(order, [f1/(0.5*fs), f2/(0.5*fs)], btype="band")
    return filtfilt(b, a, x)

def _fft_mag(x, fs):
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    mag = np.abs(np.fft.rfft(x))
    return freqs, mag

def _welch_psd(x, fs, nperseg=None):
    if nperseg is None:
        nperseg = min(256, len(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg)
    return f, Pxx

def _band_energy_psd(x, fs, f_low, f_high, nperseg=None):
    f, Pxx = _welch_psd(x, fs, nperseg=nperseg)
    m = (f >= f_low) & (f <= f_high)
    if not np.any(m):
        return 0.0
    return float(np.trapz(Pxx[m], f[m]))

def _dominant_freq_in_band_psd(x, fs, fmin, fmax, nperseg=None):
    f, Pxx = _welch_psd(x, fs, nperseg=nperseg)
    m = (f >= fmin) & (f <= fmax)
    if not np.any(m):
        return np.nan
    i = int(np.argmax(Pxx[m]))
    return float(f[m][i])

def _mad(x):
    return float(np.median(np.abs(x - np.median(x)))) + 1e-9

def _find_peaks_adaptive(x, fs, min_spacing_s=0.25, prom_scale=0.6):
    distance = max(1, int(min_spacing_s * fs))
    prom = prom_scale * _mad(x)
    idx, props = find_peaks(x, distance=distance, prominence=prom)
    return idx.astype(int), {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in props.items()}


def _estimate_gravity_vec(arr, fs, fc=0.3):
    x_lp = _butter_lowpass(arr[:,0], fs, fc=fc)
    y_lp = _butter_lowpass(arr[:,1], fs, fc=fc)
    z_lp = _butter_lowpass(arr[:,2], fs, fc=fc)

    g = np.array([np.mean(x_lp), np.mean(y_lp), np.mean(z_lp)], dtype=float)
    if np.linalg.norm(g) < 1e-6:
        g = np.array([np.mean(arr[:,0]), np.mean(arr[:,1]), np.mean(arr[:,2])], dtype=float)
    return g

def _rodrigues_R(v_from, v_to):
    a = v_from / (np.linalg.norm(v_from) + 1e-12)
    b = v_to   / (np.linalg.norm(v_to) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:  
        if c > 0.999999: 
            return np.eye(3)

        tmp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, tmp)
        axis /= (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = -np.eye(3) + 2*np.outer(axis, axis)  
        return R
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + (K @ K) * ((1 - c) / (s**2 + 1e-12))
    return R

def _align_gravity_to_plusZ(arr, fs, grav_fc=0.3):
    g_vec = _estimate_gravity_vec(arr, fs, fc=grav_fc)
    target = np.array([0.0, 0.0, np.linalg.norm(g_vec)])  # +Z with magnitude |g|
    R = _rodrigues_R(g_vec, target)
    arr_rot = (R @ arr.T).T
    info = {
        "R": R.tolist(),
        "g_vec_raw": g_vec.tolist(),
        "g_mag": float(np.linalg.norm(g_vec)),
        "aligned_to": [0.0, 0.0, float(np.linalg.norm(g_vec))]
    }
    return arr_rot, info

def _estimate_gravity_series(arr, fs, fc=0.3):
    gx = _butter_lowpass(arr[:,0], fs, fc=fc)
    gy = _butter_lowpass(arr[:,1], fs, fc=fc)
    gz = _butter_lowpass(arr[:,2], fs, fc=fc)
    return np.column_stack([gx, gy, gz])



def _butter_lowpass(x, fs, fc, order=4):
    b, a = butter(order, fc/(0.5*fs), btype="low")
    return filtfilt(b, a, x)


def _rms(x): return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))
def _p2p(x): return float(np.max(x) - np.min(x))
def _zcr(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean((x[:-1] * x[1:]) < 0))

def _butter_bandpass(x, fs, lo=0.5, hi=6.0, order=4):
    nyq = 0.5 * fs
    lo_n, hi_n = lo/nyq, hi/nyq
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)

def _welch_peak_freq(x, fs, fmin=0.3, fmax=8.0, nperseg=None, topn=3):
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), 128)
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= fmin) & (f <= fmax)
    f, pxx = f[mask], pxx[mask]
    if len(f) == 0:
        return {"dom_freq": None, "top_freqs": []}
    idx_sorted = np.argsort(pxx)[::-1]
    top_idx = idx_sorted[:topn]
    dom = float(f[top_idx[0]])
    tops = [(float(f[i]), float(pxx[i])) for i in top_idx]
    return {"dom_freq": dom, "top_freqs": tops}

def _peak_rate(x, fs, prominence=None, distance=None):

    x = np.asarray(x, dtype=float)
    if distance is None:
        distance = int(0.15 * fs) 
    peaks, _ = find_peaks(x, prominence=prominence, distance=distance)
    if len(peaks) < 2:
        return 0.0
    intervals = np.diff(peaks) / fs
    return float(1.0 / np.mean(intervals))



def _safe_sosfiltfilt(sos, x, padlen_default=0, axis=-1):
    x = np.asarray(x, dtype=float)
    N = x.shape[axis]
    if padlen_default <= 0:
        padlen_default = 3 * (2 * sos.shape[0])
    padlen = min(padlen_default, max(0, N - 1))
    if padlen <= 0:
        return x - np.mean(x, axis=axis, keepdims=True)
    return sosfiltfilt(sos, x, padlen=padlen, axis=axis)

def _butter_sos(kind, fs, fc, order=4):
    nyq = 0.5 * fs
    if kind == "low":
        wn = fc / nyq
        return butter(order, wn, btype="low", output="sos")
    if kind == "high":
        wn = fc / nyq
        return butter(order, wn, btype="high", output="sos")
    if kind == "band":
        lo, hi = fc
        wn = [lo/nyq, hi/nyq]
        return butter(order, wn, btype="band", output="sos")
    raise ValueError("kind must be low/high/band")

def _filt(x, fs, mode, fc, order=4):
    x = np.asarray(x, dtype=float)
    N = x.shape[-1] if x.ndim > 1 else x.shape[0]
    ord_use = order if N > 5 * order else max(1, order // 2)
    sos = _butter_sos(mode, fs, fc, order=ord_use)
    padlen_default = max(9, 3 * (2 * sos.shape[0]))
    return _safe_sosfiltfilt(sos, x, padlen_default=padlen_default, axis=0)



def _peak_rate_(signal, fs, distance_s=0.25, prominence=None):
    signal = np.asarray(signal, dtype=float)
    distance = max(1, int(distance_s * fs))
    peaks, props = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return 0.0, peaks
    intervals = np.diff(peaks) / fs
    rate = float(1.0 / np.mean(intervals))
    return rate, peaks

def _match_events(t_ref, t_cmp, tol_s=0.20):
    i, j = 0, 0
    matches = []
    while i < len(t_ref) and j < len(t_cmp):
        dt = t_cmp[j] - t_ref[i]
        if abs(dt) <= tol_s:
            matches.append((i, j, dt))
            i += 1; j += 1
        elif dt < -tol_s:
            j += 1
        else:
            i += 1
    return matches

def _circular_consistency(lags_s, period_s):
    if len(lags_s) == 0 or period_s <= 0:
        return 0.0
    phases = (2*np.pi * (np.asarray(lags_s)/period_s)) % (2*np.pi)
    C = np.sqrt(np.mean(np.cos(phases))**2 + np.mean(np.sin(phases))**2)
    return float(C)

def _gravity_alignment(acc, fs):
    acc = np.asarray(acc, dtype=float)
    assert acc.ndim == 2 and acc.shape[1] == 3, "acc must be Nx3"

    g = _filt(acc, fs, "low", 0.3, order=4)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    g_hat = g / g_norm  
    tmp = np.tile(np.array([1.0, 0.0, 0.0]), (acc.shape[0], 1))
    colinear = (np.abs((g_hat * tmp).sum(axis=1)) > 0.9)
    tmp[colinear] = np.array([0.0, 1.0, 0.0])

    x_body = np.cross(tmp, g_hat)
    x_norm = np.linalg.norm(x_body, axis=1, keepdims=True) + 1e-9
    x_body = x_body / x_norm

    y_body = np.cross(g_hat, x_body)  # 保证右手系


    acc_body = np.empty_like(acc)
    for n in range(acc.shape[0]):
        R = np.stack([x_body[n], y_body[n], g_hat[n]], axis=1)  # 3x3
        acc_body[n] = acc[n] @ R

    a_vert_raw = acc_body[:, 2]
    return acc_body, a_vert_raw, g_hat


def run_full_eda(arr, fs=50, step_band=(0.7, 3.0), fft_xlim=10.0,
                 unify_body_coords=True, grav_fc=0.3):

    def _to_df(a):
        a = np.asarray(a, dtype=float)
        return {
            "X": a[:,0].astype(float),
            "Y": a[:,1].astype(float),
            "Z": a[:,2].astype(float),
            "SVM": np.linalg.norm(a, axis=1).astype(float),
        }

    def _butter_lowpass(x, fs, fc=0.3, order=4):
        sos = butter(order, fc/(0.5*fs), btype="low", output="sos")
        return sosfiltfilt(sos, x)

    def _butter_highpass(x, fs, fc=0.3, order=4):
        sos = butter(order, fc/(0.5*fs), btype="high", output="sos")
        return sosfiltfilt(sos, x)

    def _butter_bandpass(x, fs, f1=0.7, f2=3.0, order=4):
        sos = butter(order, [f1/(0.5*fs), f2/(0.5*fs)], btype="band", output="sos")
        return sosfiltfilt(sos, x)

    def _welch_psd_stable(x, fs, nperseg=None):
        N = len(x)
        if nperseg is None:
            if N >= 256:
                nperseg = 256
            else:
                k = max(6, int(np.floor(np.log2(max(8, N//2)))))  # >= 64
                nperseg = 2**k
                nperseg = min(nperseg, N)
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        return f, Pxx

    def _estimate_gravity_series(a, fs, fc=0.3, order=4):
        gx = _butter_lowpass(a[:,0], fs, fc, order)
        gy = _butter_lowpass(a[:,1], fs, fc, order)
        gz = _butter_lowpass(a[:,2], fs, fc, order)
        return np.column_stack([gx, gy, gz])

    def _estimate_gravity_vec(a, fs, fc=0.3, order=4):
        g_t = _estimate_gravity_series(a, fs, fc, order)
        g = g_t.mean(axis=0)
        if np.linalg.norm(g) < 1e-9:
            g = np.array([np.mean(a[:,0]), np.mean(a[:,1]), np.mean(a[:,2])], dtype=float)
        return g

    def _rodrigues_R(v_from, v_to):
        a = v_from / (np.linalg.norm(v_from) + 1e-12)
        b = v_to   / (np.linalg.norm(v_to) + 1e-12)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if s < 1e-12:
            if c > 0.999999:
                return np.eye(3)
            tmp = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9: tmp = np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, tmp)
            axis /= (np.linalg.norm(axis) + 1e-12)
            # 180° 旋转
            return -np.eye(3) + 2*np.outer(axis, axis)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        return np.eye(3) + K + (K @ K) * ((1 - c) / (s**2 + 1e-12))

    def _align_gravity_to_plusZ(a, fs, fc=0.3):
        g_vec = _estimate_gravity_vec(a, fs, fc)
        target = np.array([0.0, 0.0, np.linalg.norm(g_vec)])
        R = _rodrigues_R(g_vec, target)
        a_rot = (R @ a.T).T
        info = {
            "R": R.tolist(),
            "g_vec_raw": g_vec.tolist(),
            "g_mag": float(np.linalg.norm(g_vec)),
            "aligned_to": [0.0, 0.0, float(np.linalg.norm(g_vec))]
        }
        return a_rot, info

    def _mad(x):
        x = np.asarray(x, dtype=float)
        return float(np.median(np.abs(x - np.median(x)))) + 1e-9

    def _find_peaks_adaptive(x, fs, min_spacing_s=0.25, prom_scale=0.6):
        x = np.asarray(x, dtype=float)
        distance = max(1, int(min_spacing_s * fs))
        rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
        prom = max(prom_scale * _mad(x), 0.4 * rms)
        idx, props = find_peaks(x, distance=distance, prominence=prom)
        if len(idx) > 12 and rms > 0:
            idx, props = find_peaks(x, distance=distance, prominence=prom, height=0.2*rms)
        props = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in props.items()}
        return idx.astype(int), props

    arr = np.asarray(arr, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 3, "arr must be Nx3"
    df  = _to_df(arr)
    N   = len(df["X"])
    t   = np.arange(N) / float(fs)

    def _stats(x):
        x = np.asarray(x, dtype=float)
        return {"mean": float(np.mean(x)),
                "std":  float(np.std(x)),
                "min":  float(np.min(x)),
                "max":  float(np.max(x)),
                "p2p":  float(np.max(x) - np.min(x))}

    stats = {k: _stats(v) for k, v in df.items()}

    body = None
    if unify_body_coords:
        arr_b, rot_info = _align_gravity_to_plusZ(arr, fs, fc=grav_fc)
        Xb, Yb, Zb = arr_b[:,0], arr_b[:,1], arr_b[:,2]
        Horiz_raw = np.sqrt(Xb**2 + Yb**2)
        Vert_raw  = Zb
        SVMb      = np.linalg.norm(arr_b, axis=1)

        g_t   = _estimate_gravity_series(arr_b, fs, fc=grav_fc)
        Xb_dyn, Yb_dyn, Zb_dyn = Xb - g_t[:,0], Yb - g_t[:,1], Zb - g_t[:,2]
        Horiz_dyn = np.sqrt(Xb_dyn**2 + Yb_dyn**2)
        Vert_dyn  = Zb_dyn

        body = {
            "Xb": Xb.tolist(),
            "Yb": Yb.tolist(),
            "Zb": Zb.tolist(),
            "Horiz": Horiz_dyn.tolist(),  
            "Vert":  Vert_dyn.tolist(),   
            "SVMb":  SVMb.tolist(),
            "rotation": rot_info,
            "summary_raw": {  
                "vert_rms":  float(np.sqrt(np.mean(Vert_raw**2))),
                "horiz_rms": float(np.sqrt(np.mean(Horiz_raw**2))),
                "vert_p2p":  float(np.max(Vert_raw) - np.min(Vert_raw)),
                "horiz_p2p": float(np.max(Horiz_raw) - np.min(Horiz_raw)),
            },
            "summary": {    
                "vert_rms":  float(np.sqrt(np.mean(Vert_dyn**2))),
                "horiz_rms": float(np.sqrt(np.mean(Horiz_dyn**2))),
                "vert_p2p":  float(np.max(Vert_dyn) - np.min(Vert_dyn)),
                "horiz_p2p": float(np.max(Horiz_dyn) - np.min(Horiz_dyn)),
            },
            "dyn_signals": {
                "Xb_dyn": Xb_dyn.tolist(),
                "Yb_dyn": Yb_dyn.tolist(),
                "Zb_dyn": Zb_dyn.tolist(),
            },
        }
        Vert_dyn_for_gait = Vert_dyn
        a_dyn = np.column_stack([Xb_dyn, Yb_dyn, Zb_dyn])
    else:
        g_t  = _estimate_gravity_series(arr, fs, fc=grav_fc)
        a_dyn = arr - g_t

        g_hat = g_t.mean(axis=0); g_hat /= (np.linalg.norm(g_hat) + 1e-12)
        Vert_dyn_for_gait = a_dyn @ g_hat


    X_hp   = _butter_highpass(df["X"], fs, 0.3)
    Y_hp   = _butter_highpass(df["Y"], fs, 0.3)
    Z_hp   = _butter_highpass(df["Z"], fs, 0.3)
    SVM_hp = _butter_highpass(df["SVM"], fs, 0.3)

    X_bp   = _butter_bandpass(df["X"], fs, step_band[0], step_band[1])
    Y_bp   = _butter_bandpass(df["Y"], fs, step_band[0], step_band[1])
    Z_bp   = _butter_bandpass(df["Z"], fs, step_band[0], step_band[1])
    SVM_bp = _butter_bandpass(df["SVM"], fs, step_band[0], step_band[1])


    freqs = np.fft.rfftfreq(len(SVM_hp), d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(SVM_hp))
    psd_f, psd_Pxx = _welch_psd_stable(SVM_hp, fs)


    vert_bp = _butter_bandpass(Vert_dyn_for_gait, fs, step_band[0], step_band[1])
    f_psd, Pxx_psd = _welch_psd_stable(vert_bp, fs, nperseg=min(256, len(vert_bp)))
    mband = (f_psd >= step_band[0]) & (f_psd <= step_band[1])
    if np.any(mband):
        f_slice = f_psd[mband]; P_slice = Pxx_psd[mband]
        dom_f = float(f_slice[np.argmax(P_slice)])
    else:
        dom_f = float("nan")
    dom_spm = float(dom_f * 60.0) if np.isfinite(dom_f) else float("nan")


    if np.isfinite(dom_f) and dom_f > 0:
        f0 = dom_f
        low_win  = (max(0.2, f0 - 0.25), f0 + 0.25)
        high_win = (f0 + 0.4, 5.0)

        def _band_energy_sig(x, fs, f_low, f_high):
            f, Pxx = _welch_psd_stable(x, fs, nperseg=min(256, len(x)))
            m = (f >= f_low) & (f <= f_high)
            if not np.any(m): return float("nan")
            return float(trapezoid(Pxx[m], f[m]))

        low_E  = _band_energy_sig(Vert_dyn_for_gait, fs, *low_win)
        high_E = _band_energy_sig(Vert_dyn_for_gait, fs, *high_win)
        low_high_energy_ratio = (low_E / (high_E + 1e-9)) if np.isfinite(low_E) and np.isfinite(high_E) else float("nan")

        def _band_pow_from_psd(fv, Pv, fc, tol=0.2):
            m = (fv >= fc - tol) & (fv <= fc + tol)
            if not np.any(m): return float("nan")
            return float(trapezoid(Pv[m], fv[m]))

        base  = _band_pow_from_psd(f_psd, Pxx_psd, f0)
        harm2 = _band_pow_from_psd(f_psd, Pxx_psd, 2.0*f0)
        harmonic_2x_ratio = (harm2 / (base + 1e-9)) if np.isfinite(base) and base > 0 else float("nan")
    else:
        low_high_energy_ratio = float("nan")
        harmonic_2x_ratio = float("nan")

    freq_summary = {
        "dom_freq_hz": dom_f,
        "dom_freq_spm": dom_spm,
        "low_high_energy_ratio": low_high_energy_ratio,
        "harmonic_2x_ratio": harmonic_2x_ratio
    }


    peaks_idx, peaks_props = _find_peaks_adaptive(vert_bp, fs, min_spacing_s=0.25, prom_scale=0.6)
    if len(peaks_idx) >= 2:
        ipi = np.diff(peaks_idx) / fs
        ipi = ipi[ipi > 0]
        if len(ipi) > 0:
            cadence_spm = float(60.0 / np.mean(ipi))
            ibi_mean_s  = float(np.mean(ipi))
            ibi_std_s   = float(np.std(ipi))
            ibi_cv      = float(ibi_std_s / (ibi_mean_s + 1e-12))
        else:
            cadence_spm, ibi_mean_s, ibi_std_s, ibi_cv = (float("nan"),)*4
    else:
        cadence_spm, ibi_mean_s, ibi_std_s, ibi_cv = (float("nan"),)*4


    out = {
        "meta": {"N": int(N), "fs": float(fs)},
        "stats": stats,
        "time": {"t": t.tolist()},
        "raw": {k: v.tolist() for k, v in df.items()},  # X,Y,Z,SVM
        "hp":  {"X_hp": X_hp.tolist(), "Y_hp": Y_hp.tolist(), "Z_hp": Z_hp.tolist(), "SVM_hp": SVM_hp.tolist()},
        "bp":  {"X_bp": X_bp.tolist(), "Y_bp": Y_bp.tolist(), "Z_bp": Z_bp.tolist(), "SVM_bp": SVM_bp.tolist()},
        "fft": {"freqs": freqs.tolist(), "mag": fft_mag.tolist(), "xlim": float(fft_xlim)},
        "psd": {"f": psd_f.tolist(), "Pxx": psd_Pxx.tolist()},
        "peaks": {"indices": peaks_idx.tolist(), "props": peaks_props},
        "cadence": {"from_peaks_spm": float(cadence_spm)},
        "freq_summary": freq_summary
    }
    if body is not None:
        out["body"] = body


    if a_dyn is None or a_dyn.shape[0] < 3 or not np.isfinite(fs):
        out["jerk_summary"] = {"rms": float("nan"), "p95": float("nan"),
                               "vert_rms": float("nan"), "vert_p95": float("nan")}
    else:
        Np = a_dyn.shape[0]
        win = 11; poly = 2
        if win % 2 == 0: win += 1
        if win > Np: win = Np - (1 - Np % 2)
        if win >= 5:
            a_dyn_f = savgol_filter(a_dyn, window_length=win, polyorder=min(poly, win-1), axis=0)
            vert_dyn_f = savgol_filter(Vert_dyn_for_gait, window_length=win, polyorder=min(poly, win-1))
        else:
            a_dyn_f = a_dyn
            vert_dyn_f = Vert_dyn_for_gait

        jerk_vec  = np.diff(a_dyn_f, axis=0) * fs
        jerk_mag  = np.linalg.norm(jerk_vec, axis=1)
        jerk_vert = np.diff(vert_dyn_f) * fs

        out["jerk_summary"] = {
            "rms": float(np.sqrt(np.mean(jerk_mag**2))) if jerk_mag.size else float("nan"),
            "p95": float(np.percentile(jerk_mag, 95))    if jerk_mag.size else float("nan"),
            "vert_rms": float(np.sqrt(np.mean(jerk_vert**2))) if jerk_vert.size else float("nan"),
            "vert_p95": float(np.percentile(np.abs(jerk_vert), 95)) if jerk_vert.size else float("nan"),
        }

    return out


def jerk_summary_from_arr(
    arr, fs=50.0, grav_fc=0.3,
    smooth="savgol", win=11, poly=2
):

    arr = np.asarray(arr, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 3, "arr must be Nx3"
    N = arr.shape[0]
    if N < 3 or not np.isfinite(fs):
        return {"rms": np.nan, "p95": np.nan, "vert_rms": np.nan, "vert_p95": np.nan}


    g_t   = _estimate_gravity_series(arr, fs, fc=grav_fc)  # Nx3
    a_dyn = arr - g_t                                      # Nx3
    g_hat = g_t.mean(axis=0)
    g_hat = g_hat / (np.linalg.norm(g_hat) + 1e-12)
    vert_dyn = a_dyn @ g_hat
    if smooth == "savgol" and N >= 5:
        win = int(win)
        if win % 2 == 0: win += 1
        if win > N: win = N - (1 - N % 2)  
        if win >= 5:
            a_dyn_f   = savgol_filter(a_dyn, window_length=win, polyorder=min(poly, win-1), axis=0)
            vert_dyn_f= savgol_filter(vert_dyn, window_length=win, polyorder=min(poly, win-1))
        else:
            a_dyn_f, vert_dyn_f = a_dyn, vert_dyn
    else:
        a_dyn_f, vert_dyn_f = a_dyn, vert_dyn

    jerk_vec = np.diff(a_dyn_f, axis=0) * fs          # (N-1, 3)
    jerk_mag = np.linalg.norm(jerk_vec, axis=1)       # (N-1,)
    jerk_vert = np.diff(vert_dyn_f) * fs              # (N-1,)

    out = {
        "rms": float(np.sqrt(np.mean(jerk_mag**2))) if jerk_mag.size else np.nan,
        "p95": float(np.percentile(jerk_mag, 95))    if jerk_mag.size else np.nan,
        "vert_rms": float(np.sqrt(np.mean(jerk_vert**2))) if jerk_vert.size else np.nan,
        "vert_p95": float(np.percentile(np.abs(jerk_vert), 95)) if jerk_vert.size else np.nan,
    }
    return out


def simplify_eda_output(eda, include_jerk=True):
    out = {}
    out["meta"]  = dict(eda.get("meta", {}))
    out["stats"] = dict(eda.get("stats", {}))

    body_sum = {}
    try:
        body_sum_src = eda.get("body", {}).get("summary", {})
        body_sum = {
            "vert_rms":  float(body_sum_src.get("vert_rms",  np.nan)),
            "horiz_rms": float(body_sum_src.get("horiz_rms", np.nan)),
            "vert_p2p":  float(body_sum_src.get("vert_p2p",  np.nan)),
            "horiz_p2p": float(body_sum_src.get("horiz_p2p", np.nan)),
        }
    except Exception:
        body_sum = {}
    out["body_summary"] = body_sum

    fsu = eda.get("freq_summary", {})
    out["freq_summary"] = {
        "dom_freq_hz":           float(fsu.get("dom_freq_hz", np.nan)),
        "dom_freq_spm":          float(fsu.get("dom_freq_spm", np.nan)),
        "low_high_energy_ratio": float(fsu.get("low_high_energy_ratio", np.nan)),
        "harmonic_2x_ratio":     float(fsu.get("harmonic_2x_ratio", np.nan)),
    }

    ps_up = eda.get("peaks_summary", {})
    if {"ibi_mean_s","ibi_std_s","ibi_cv"}.issubset(ps_up.keys()):
        out["peaks_summary"] = {
            "peak_count": int(ps_up.get("peak_count", 0)),
            "ibi_mean_s": float(ps_up.get("ibi_mean_s", np.nan)),
            "ibi_std_s":  float(ps_up.get("ibi_std_s",  np.nan)),
            "ibi_cv":     float(ps_up.get("ibi_cv",     np.nan)),
        }
    else:
        fs = float(eda.get("meta", {}).get("fs", np.nan))
        peak_idx = np.asarray(eda.get("peaks", {}).get("indices", []), dtype=float)
        ibi_mean = ibi_std = ibi_cv = np.nan
        if np.isfinite(fs) and peak_idx.size >= 2:
            ipi = np.diff(peak_idx) / fs
            ipi = ipi[ipi > 0]
            if ipi.size > 0:
                ibi_mean = float(np.mean(ipi))
                ibi_std  = float(np.std(ipi))
                ibi_cv   = float(ibi_std / (ibi_mean + 1e-12))
        out["peaks_summary"] = {
            "peak_count": int(peak_idx.size),
            "ibi_mean_s": ibi_mean,
            "ibi_std_s":  ibi_std,
            "ibi_cv":     ibi_cv,
        }

    if include_jerk:
        if "jerk_summary" in eda and isinstance(eda["jerk_summary"], dict):
            js = eda["jerk_summary"]
            out["jerk_summary"] = {
                "rms": float(js.get("rms", np.nan)),
                "p95": float(js.get("p95", np.nan)),
                "vert_rms": float(js.get("vert_rms", np.nan)),
                "vert_p95": float(js.get("vert_p95", np.nan)),
            }
        else:
            fs = float(eda.get("meta", {}).get("fs", np.nan))
            jerk_rms = jerk_p95 = np.nan
            try:
                dyn_sig = eda.get("body", {}).get("dyn_signals", {})
                Xb_dyn = np.asarray(dyn_sig.get("Xb_dyn", []), dtype=float)
                Yb_dyn = np.asarray(dyn_sig.get("Yb_dyn", []), dtype=float)
                Zb_dyn = np.asarray(dyn_sig.get("Zb_dyn", []), dtype=float)
                if Xb_dyn.size >= 3 and Xb_dyn.size == Yb_dyn.size == Zb_dyn.size and np.isfinite(fs):
                    a_dyn = np.column_stack([Xb_dyn, Yb_dyn, Zb_dyn])
                    jerk_vec = np.diff(a_dyn, axis=0) * fs
                    jerk_mag = np.linalg.norm(jerk_vec, axis=1)
                    jerk_rms = float(np.sqrt(np.mean(jerk_mag**2)))
                    jerk_p95 = float(np.percentile(jerk_mag, 95))
                else:
                    svm_hp = np.asarray(eda.get("hp", {}).get("SVM_hp", []), dtype=float)
                    if svm_hp.size >= 3 and np.isfinite(fs):
                        jerk = np.diff(svm_hp) * fs
                        absj = np.abs(jerk)
                        jerk_rms = float(np.sqrt(np.mean(absj**2)))
                        jerk_p95 = float(np.percentile(absj, 95))
            except Exception:
                pass
            out["jerk_summary"] = {"rms": jerk_rms, "p95": jerk_p95}

    return out




def run_eda_slim(arr, fs=50, step_band=(0.7, 3.0), fft_xlim=10.0, unify_body_coords=True, grav_fc=0.3, include_jerk=True):
    eda_full = run_full_eda(
        arr, fs=fs, step_band=step_band, fft_xlim=fft_xlim,
        unify_body_coords=unify_body_coords, grav_fc=grav_fc
    )
    return simplify_eda_output(eda_full, include_jerk=include_jerk)


def extract_gyro_features(arr, fs=50):
    arr = np.asarray(arr, dtype=float)
    assert arr.ndim == 2 and arr.shape[1] == 3, "arr must be Nx3 (ωx, ωy, ωz)"
    wx, wy, wz = arr[:,0], arr[:,1], arr[:,2]
    N = len(wx)
    dur = float(N / fs)

    mean = (float(np.mean(wx)), float(np.mean(wy)), float(np.mean(wz)))
    rms  = (_rms(wx), _rms(wy), _rms(wz))
    p2p  = (_p2p(wx), _p2p(wy), _p2p(wz))
    zcr  = (_zcr(wx), _zcr(wy), _zcr(wz))

    wmag = np.linalg.norm(arr, axis=1)
    wmag_mean, wmag_rms, wmag_p2p = float(np.mean(wmag)), _rms(wmag), _p2p(wmag)

    E = np.sum(arr**2, axis=0)  
    Etot = float(np.sum(E)) + 1e-12
    energy_frac = tuple(float(e / Etot) for e in E) 

    dt = 1.0 / fs
    net_angle = (float(np.sum(wx)*dt), float(np.sum(wy)*dt), float(np.sum(wz)*dt))
    abs_angle = (float(np.sum(np.abs(wx))*dt),
                 float(np.sum(np.abs(wy))*dt),
                 float(np.sum(np.abs(wz))*dt))


    welch_mag = _welch_peak_freq(wmag, fs=fs, fmin=0.3, fmax=8.0, topn=5)
    welch_x   = _welch_peak_freq(wx,   fs=fs, fmin=0.3, fmax=8.0, topn=5)


    wx_bp = _butter_bandpass(wx, fs, lo=0.5, hi=6.0, order=4)
    wmag_bp = _butter_bandpass(wmag, fs, lo=0.5, hi=6.0, order=4)

    peak_rate_x   = _peak_rate(wx_bp,   fs, prominence=None, distance=int(0.15*fs))
    peak_rate_mag = _peak_rate(wmag_bp, fs, prominence=None, distance=int(0.15*fs))

    step_freq_est = float(0.5 * peak_rate_mag) 

    features = {
        "meta": {
            "N": int(N),
            "fs": float(fs),
            "duration_s": dur,
        },
        "time_stats": {
            "mean_xyz": mean,
            "rms_xyz": rms,
            "p2p_xyz": p2p,
            "zcr_xyz": zcr,
            "wmag_mean": wmag_mean,
            "wmag_rms": wmag_rms,
            "wmag_p2p": wmag_p2p,
        },
        "energy": {
            "energy_frac_xyz": energy_frac,  
            "net_angle_xyz_rad": net_angle,  
            "abs_angle_xyz_rad": abs_angle,   
        },
        "spectral": {
            "welch_wmag": {
                "dom_freq_hz": welch_mag["dom_freq"],   
                "top_freqs": welch_mag["top_freqs"],    
            },
            "welch_wx": {
                "dom_freq_hz": welch_x["dom_freq"],
                "top_freqs": welch_x["top_freqs"],
            },
            "peak_rate_hz": {
                "wx_bandpassed": peak_rate_x,    
                "wmag_bandpassed": peak_rate_mag,
            },
            "step_freq_est_hz": step_freq_est,  
        }
    }
    return features



def gait_sync_and_impact(acc, gyro, fs=50.0):
    acc = np.asarray(acc, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    assert acc.shape == gyro.shape and acc.ndim == 2 and acc.shape[1] == 3, "acc & gyro must be Nx3"

    N = acc.shape[0]; T = N / fs; t = np.arange(N) / fs


    amag = np.linalg.norm(acc, axis=1)
    wmag = np.linalg.norm(gyro, axis=1)
    amag_bp = _filt(amag, fs, "band", (0.7, 3.0), order=4)
    wmag_bp = _filt(wmag, fs, "band", (0.7, 3.0), order=4)
    acc_body, a_vert_raw, g_hat = _gravity_alignment(acc, fs)
    a_vert_dyn = _filt(a_vert_raw, fs, "high", 0.5, order=4) 
    a_horiz_mag = np.sqrt(acc_body[:,0]**2 + acc_body[:,1]**2)

    vert_bp = _filt(a_vert_dyn, fs, "band", (0.7, 3.0), order=4)
    horiz_bp = _filt(a_horiz_mag, fs, "band", (0.7, 3.0), order=4)
    step_rate_acc, peaks_acc = _peak_rate_(amag_bp, fs, distance_s=0.25, prominence=None)
    step_rate_gyro, peaks_gyro = _peak_rate_(wmag_bp, fs, distance_s=0.25, prominence=None)
    step_times_acc = t[peaks_acc]; step_times_gyro = t[peaks_gyro]
    step_rate_vert, _ = _peak_rate_(vert_bp, fs, distance_s=0.25, prominence=None)
    step_rate_horiz, _ = _peak_rate_(horiz_bp, fs, distance_s=0.25, prominence=None)
    matches = _match_events(step_times_acc, step_times_gyro, tol_s=0.20)
    lags = [dt for (_, _, dt) in matches]
    mean_lag = float(np.mean(lags)) if lags else 0.0
    median_abs_lag = float(np.median(np.abs(lags))) if lags else 0.0
    period = (np.median(np.diff(step_times_acc)) if len(step_times_acc) >= 2
              else (1.0/step_rate_acc if step_rate_acc > 0 else 0.0))
    phase_consistency = _circular_consistency(lags, period)

    impacts = []
    win = int(0.20 * fs)  # +/-200 ms
    for idx in peaks_acc:
        i0, i1 = max(0, idx-win), min(N, idx+win+1)
        seg = a_vert_dyn[i0:i1]
        if len(seg) > 3:
            impacts.append(float(np.max(seg) - np.min(seg)))
    impact_median = float(np.median(impacts)) if impacts else 0.0
    impact_p95 = float(np.percentile(impacts, 95)) if impacts else 0.0

    summary = {
        "meta": {"N": int(N), "fs": float(fs), "duration_s": float(T)},
        "step_metrics": {
            "step_rate_acc_hz": float(step_rate_acc),
            "step_rate_gyro_hz": float(step_rate_gyro),
            "step_rate_diff_hz": float(step_rate_acc - step_rate_gyro),
            "n_steps_acc": int(len(peaks_acc)),
            "n_steps_gyro": int(len(peaks_gyro)),
            "step_rate_vert_hz": float(step_rate_vert),
            "step_rate_horiz_hz": float(step_rate_horiz),
        },
        "sync_metrics": {
            "n_matched_pairs": int(len(matches)),
            "mean_lag_s": mean_lag,
            "median_abs_lag_s": median_abs_lag,
            "phase_consistency_0to1": float(phase_consistency),
        },
        "vertical_impact": {
            "per_step_peak_to_valley_mps2": impacts,
            "impact_median_mps2": impact_median,
            "impact_p95_mps2": impact_p95,
        },
        "body_frame": {
            "vert_rms_mps2": _rms(a_vert_dyn),
            "horiz_rms_mps2": _rms(a_horiz_mag),
            "vert_p2p_mps2": float(np.max(a_vert_dyn) - np.min(a_vert_dyn)) if N>0 else 0.0,
        }
    }
    return summary
