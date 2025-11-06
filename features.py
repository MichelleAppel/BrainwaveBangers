import numpy as np
from scipy.signal import welch, butter, filtfilt
from config import FS, ALPHA, THETA, BETA, CH_NAMES, FZ_SPIKE_UV

def bandpower_1d(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256))
    idx = (f >= fmin) & (f <= fmax)
    return float(Pxx[idx].mean()) if np.any(idx) else 0.0

HP = butter(2, 1.0 / (FS / 2), btype="highpass")
LP = butter(2, 40.0 / (FS / 2), btype="lowpass")

def prefilter_window(win):
    for i in range(win.shape[0]):
        x = filtfilt(*HP, win[i])
        x = filtfilt(*LP, x)
        win[i] = x
    return win

def compute_features(window):
    """Return arousal_raw, valence_raw, artifact_flag."""
    window = prefilter_window(window.copy())

    fz = window[CH_NAMES.index("Fz")]
    artifact = int(np.max(np.abs(fz)) > FZ_SPIKE_UV)

    alpha = np.array([bandpower_1d(window[i], FS, *ALPHA) for i in range(len(CH_NAMES))])
    theta = np.array([bandpower_1d(window[i], FS, *THETA) for i in range(len(CH_NAMES))])
    beta  = np.array([bandpower_1d(window[i], FS, *BETA ) for i in range(len(CH_NAMES))])

    pick = [CH_NAMES.index(n) for n in ["F3","F4","Cz"]]
    a_bar = alpha[pick].mean(); t_bar = theta[pick].mean(); b_bar = beta[pick].mean()

    eps = 1e-12
    arousal_raw = np.log(b_bar + eps) - np.log(a_bar + t_bar + eps)
    f3 = CH_NAMES.index("F3"); f4 = CH_NAMES.index("F4")
    valence_raw = np.log(alpha[f3] + eps) - np.log(alpha[f4] + eps)

    return float(arousal_raw), float(valence_raw), artifact

def bandpower_1d(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 256))
    idx = (f >= fmin) & (f <= fmax)
    return float(Pxx[idx].mean()) if np.any(idx) else 0.0

def compute_bandpowers(window):
    """
    Returns dict of mean bandpowers over BAND_PICKS
    e.g. {"delta": 0.12, "theta": 0.08, "alpha": 0.05, "beta": 0.03, "gamma": 0.01}
    """
    pick_idx = [CH_NAMES.index(n) for n in BAND_PICKS]
    bands = {}
    for name, (fmin, fmax) in BAND_RANGES.items():
        vals = [bandpower_1d(window[i], FS, fmin, fmax) for i in pick_idx]
        bands[name] = float(np.mean(vals))
    return bands