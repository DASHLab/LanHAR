from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt




    
def _safe_sosfiltfilt(sos, x, axis=-1, padlen_default=0):
    x = np.asarray(x, dtype=float)
    N = x.shape[axis]
    if padlen_default <= 0:
        padlen_default = 3 * (2 * sos.shape[0])
    padlen = min(padlen_default, max(0, N - 1))
    if padlen <= 0:
        return x - np.mean(x, axis=axis, keepdims=True)
    return sosfiltfilt(sos, x, padlen=padlen, axis=axis)


def _butter_sos(kind: str, fs: float, fc, order=4):
    nyq = 0.5 * fs
    if kind == 'low':
        wn = float(fc) / nyq
        return butter(order, wn, btype='low', output='sos')
    elif kind == 'high':
        wn = float(fc) / nyq
        return butter(order, wn, btype='high', output='sos')
    elif kind == 'band':
        lo, hi = fc
        wn = (float(lo) / nyq, float(hi) / nyq)
        return butter(order, wn, btype='band', output='sos')
    else:
        raise ValueError('kind must be low/high/band')


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _rotmat_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps vector a -> vector b (both 3d)."""
    a = _unit(a)
    b = _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        # a || b (or opposite)
        if c > 0:  # same direction
            return np.eye(3)
        axis = _unit(np.cross(a, np.array([1,0,0])) or np.cross(a, np.array([0,1,0])))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return -np.eye(3) + 2*np.outer(axis, axis)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s**2))
    return R


def _winsorize(x: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo = np.quantile(x, lo_q)
    hi = np.quantile(x, hi_q)
    return np.clip(x, lo, hi)


def _vector_winsorize(X: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    Xw = X.copy()
    for j in range(X.shape[1]):
        Xw[:, j] = _winsorize(X[:, j], lo_q, hi_q)
    return Xw



def _estimate_gravity(arr_xyz: np.ndarray, fs: float, lp_fc=0.30, use_mean_if_short=True):
    arr_xyz = np.asarray(arr_xyz, dtype=float)
    N = arr_xyz.shape[0]
    if use_mean_if_short and N < int(fs * 1.2):  # < ~1.2 s window
        g_vec_lp = np.mean(arr_xyz, axis=0)
    else:
        sos_lp = _butter_sos('low', fs, lp_fc, order=2)
        g_vec_lp = np.mean(_safe_sosfiltfilt(sos_lp, arr_xyz, axis=0), axis=0)
    g_mag_lp = float(np.linalg.norm(g_vec_lp))
    g_vec_raw = np.mean(arr_xyz, axis=0)
    g_mag_raw = float(np.linalg.norm(g_vec_raw))

    ax = np.abs(g_vec_lp)
    dominant_axis = int(np.argmax(ax))
    axis_ratio = float(ax[dominant_axis] / max(1e-6, np.sort(ax)[-2]))  # top / second

    return {
        'g_vec_raw': g_vec_raw,
        'g_mag_raw': g_mag_raw,
        'g_vec_lp': g_vec_lp,
        'g_mag_lp': g_mag_lp,
        'quality': {'axis_ratio': axis_ratio, 'dominant_axis': dominant_axis}
    }


def _align_to_plusZ(arr_xyz: np.ndarray, g_vec: np.ndarray):
    R = _rotmat_from_to(g_vec, np.array([0.0, 0.0, 1.0]))
    aligned = arr_xyz @ R.T
    vert = aligned[:, 2]
    horiz_xy = aligned[:, :2]
    horiz_mag = np.sqrt(np.sum(horiz_xy**2, axis=1))
    return R, aligned, vert, horiz_xy, horiz_mag



def preprocess_acc_segment(arr, fs: float = 50.0, mode: str = "auto", config: dict | None = None):

    X = np.asarray(arr, dtype=float)
    assert X.ndim == 2 and X.shape[1] == 3, "arr must be (N,3) accelerometer"
    N = X.shape[0]


    cfg = {
        'gravity_lp_fc': 0.30,          
        'hp_fc': 0.30,           
        'bp_lo': 0.70, 'bp_hi': 3.00,   
        'hp_order': 2, 'bp_order': 4,
        'winsor_lo': 0.01, 'winsor_hi': 0.99,
        'winsor_lo_strong': 0.005, 'winsor_hi_strong': 0.995,
        'saturation_g': 20.0,       
        'saturation_frac_flag': 0.02,   
    }
    if config:
        cfg.update(config)


    mode = (mode or 'auto').lower()
    if mode == 'seg3_like':
        winsor_lo, winsor_hi = cfg['winsor_lo_strong'], cfg['winsor_hi_strong']
    else:
        winsor_lo, winsor_hi = cfg['winsor_lo'], cfg['winsor_hi']

    if mode == 'auto':

        med_abs = float(np.median(np.linalg.norm(X, axis=1)))
        max_abs = float(np.max(np.linalg.norm(X, axis=1)))

        g_est = _estimate_gravity(X, fs, lp_fc=cfg['gravity_lp_fc'])
        axis_ratio = g_est['quality']['axis_ratio']
        if max_abs > cfg['saturation_g'] * 0.85 or med_abs > 12.0:
            mode_eff = 'seg3_like'  
        elif axis_ratio < 1.15:
            mode_eff = 'seg2_like'  
        else:

            dom = g_est['quality']['dominant_axis']
            mode_eff = 'seg4_like' if dom == 0 else 'seg1_like'
    else:
        mode_eff = mode


    X_w = X.copy()
    if mode_eff == 'seg3_like':
        X_w = _vector_winsorize(X_w, winsor_lo, winsor_hi)


    g_est = _estimate_gravity(X_w, fs, lp_fc=cfg['gravity_lp_fc'])
    R, aligned, vert, horiz_xy, horiz_mag = _align_to_plusZ(X_w, g_est['g_vec_lp'])


    sos_hp = _butter_sos('high', fs, cfg['hp_fc'], order=cfg['hp_order'])
    sos_bp = _butter_sos('band', fs, (cfg['bp_lo'], cfg['bp_hi']), order=cfg['bp_order'])

    vert_hp = _safe_sosfiltfilt(sos_hp, vert)
    horiz_mag_hp = _safe_sosfiltfilt(sos_hp, horiz_mag)

    vert_bp = _safe_sosfiltfilt(sos_bp, vert_hp)
    horiz_mag_bp = _safe_sosfiltfilt(sos_bp, horiz_mag_hp)


    norms = np.linalg.norm(X, axis=1)
    sat_mask = norms > cfg['saturation_g']
    outlier_mask = np.any((X < np.quantile(X, winsor_lo, axis=0)) |
                          (X > np.quantile(X, winsor_hi, axis=0)), axis=1)
    possible_saturation = (np.mean(sat_mask) > cfg['saturation_frac_flag'])

    out = {
        'meta': {'fs': fs, 'N': N, 'mode_in': mode, 'mode_eff': mode_eff},
        'gravity': {
            'g_vec_raw': g_est['g_vec_raw'],
            'g_mag_raw': g_est['g_mag_raw'],
            'g_vec_lp': g_est['g_vec_lp'],
            'g_mag_lp': g_est['g_mag_lp'],
            'rot_R': R,
            'quality': g_est['quality'],
        },
        'aligned': {
            'acc_xyz': aligned,
            'vert': vert,
            'horiz_xy': horiz_xy,
            'horiz_mag': horiz_mag,
        },
        'filtered': {
            'vert_hp': vert_hp,
            'horiz_mag_hp': horiz_mag_hp,
            'vert_bp': vert_bp,
            'horiz_mag_bp': horiz_mag_bp,
            'sos_hp': sos_hp,
            'sos_bp': sos_bp,
        },
        'robust': {
            'winsor_limits': (winsor_lo, winsor_hi),
            'outlier_mask': outlier_mask,
            'saturation_mask': sat_mask,
            'possible_saturation': bool(possible_saturation),
        },
        'params_used': {
            **cfg,
            'mode_eff': mode_eff,
        }
    }
    return out

