import math
from config import TARGET, K_SCALE
from scipy.signal import butter, filtfilt

def bandpass_filter(x, fs, low=1.0, high=20.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x, axis=-1)

def ema(prev, x, alpha):
    if prev is None:
        return x
    return (1 - alpha) * prev + alpha * x

def distance(A, V):
    return math.sqrt((A - TARGET["A_star"])**2 + (V - TARGET["V_star"])**2)

def squash_tanh(x, m, s, k=K_SCALE):
    z = (x - m) / (k * s + 1e-9)
    return 0.5 + 0.5 * math.tanh(z)
