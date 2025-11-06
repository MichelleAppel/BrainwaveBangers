import numpy as np
from config import CH_NAMES, FS
# Try to import Unicorn; we’ll fallback gracefully if unavailable.
try:
    import UnicornPy
    HAVE_UNICORN = True
except Exception:
    HAVE_UNICORN = False

class UnicornSource:
    def __init__(self):
        devs = UnicornPy.GetAvailableDevices(True)
        if not devs:
            raise RuntimeError("No Unicorn devices found.")
        self.name = devs[0]
        self.dev = UnicornPy.Unicorn(self.name)
        self.n_ch = self.dev.GetNumberOfAcquiredChannels()
        self.dev.StartAcquisition(False)
        print("Connected:", self.name, "with", self.n_ch, "channels.")

    def pull_chunk(self, n_samples, frame_len=8):
        out = np.zeros((self.n_ch, n_samples), dtype=np.float32)
        got = 0
        while got < n_samples:
            need = min(frame_len, n_samples - got)
            buf_bytes = bytearray(need * self.n_ch * 4)
            self.dev.GetData(need, buf_bytes, len(buf_bytes))
            arr = np.frombuffer(buf_bytes, dtype=np.float32).reshape((need, self.n_ch)).T
            out[:, got:got+need] = arr
            got += need
        return out[:len(CH_NAMES), :]

    def set_current_params(self, params_dict):
        pass

    def close(self):
        try:
            self.dev.StopAcquisition()
        finally:
            del self.dev
            print("Device closed.")

class MockEEGSource:
    """EEG-like signal; used only in mock EEG mode when not using the oracle."""
    def __init__(self, fs=FS, n_ch=len(CH_NAMES)):
        self.fs = fs
        self.n_ch = n_ch
        self.t = 0
        self._bias_alpha = 0.0
        self._bias_beta = 0.0
        rng = np.random.default_rng(42)
        self.phase = rng.uniform(0, 2*np.pi, size=(n_ch, 3))
        self.alpha_base = rng.uniform(0.6, 1.0, size=n_ch)
        self.beta_base  = rng.uniform(0.4, 0.9, size=n_ch)
        self.theta_base = rng.uniform(0.5, 1.0, size=n_ch)
        print("Mock EEG source active.")

    def set_current_params(self, params_dict):
        d = params_dict.get("drums", 0.5)
        p = params_dict.get("pad",   0.5)
        t = params_dict.get("tempo", 0.5)
        g = params_dict.get("grain", 0.5)
        self._bias_beta  = 0.35*(d-0.5) + 0.35*(t-0.5) + 0.10*(g-0.5)
        self._bias_alpha = 0.35*(p-0.5) - 0.08*(g-0.5)

    def pull_chunk(self, n_samples, frame_len=8):
        ts = (self.t + np.arange(n_samples)) / self.fs
        self.t += n_samples
        ftheta, falpha, fbeta = 6.0, 10.0, 20.0
        theta = np.sin(2*np.pi*ftheta*ts)[None,:]
        alpha = np.sin(2*np.pi*falpha*ts)[None,:]
        beta  = np.sin(2*np.pi*fbeta *ts)[None,:]

        slow = np.sin(2*np.pi*0.05*ts)[None,:]
        out = np.zeros((self.n_ch, n_samples), dtype=np.float32)
        for ch in range(self.n_ch):
            th = self.theta_base[ch]*(theta * np.cos(self.phase[ch,0]))
            al = (self.alpha_base[ch] + self._bias_alpha)*(alpha * np.cos(self.phase[ch,1]))
            be = (self.beta_base[ch]  + self._bias_beta )*(beta  * np.cos(self.phase[ch,2]))
            signal = 15.0*(th + 1.2*al + 0.8*be)
            signal += 5.0*slow
            noise = np.random.normal(0, 3.0, size=(1, n_samples))
            out[ch] = (signal + noise).astype(np.float32)

        if np.random.rand() < 0.02:
            idx = CH_NAMES.index("Fz")
            pos = np.random.randint(0, n_samples)
            out[idx, pos:pos+1] += 1500.0
        return out

    def close(self):
        print("Mock EEG closed.")

def mock_av_response(params, noise_sigma=0.05):
    """Causal, smooth mapping + small noise."""
    d = float(params["drums"])
    p = float(params["pad"])
    t = float(params["tempo"])
    g = float(params["grain"])
    arousal = 0.6*d + 0.9*t - 0.2*p - 0.1*g + np.random.normal(0, noise_sigma)
    valence = 0.5*p + 0.2*t - 0.3*d + 0.1*g + np.random.normal(0, 0.05)
    return float(np.clip(arousal, 0.0, 1.0)), float(np.clip(valence, 0.0, 1.0))
