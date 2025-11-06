# plotting.py
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class LivePlot:
    """
    Two-panel live plot:
      - Top: Arousal & Valence (avg) with optional smoothing overlays + target lines
      - Bottom: Music parameters (drums, pad, tempo, grain)
    """
    def __init__(self, A_star=None, V_star=None, smooth_window=5):
        """
        Args:
            A_star (float|None): horizontal target line for arousal.
            V_star (float|None): horizontal target line for valence.
            smooth_window (int): moving-average window for visual smoothing (>=1 disables overlay).
        """
        self.times = []
        self.A_vals, self.V_vals = [], []
        self.param_hist = {k: [] for k in ["drums", "pad", "tempo", "grain"]}

        self.A_star, self.V_star = A_star, V_star
        self.smooth_window = max(1, int(smooth_window))
        self._Aq = deque(maxlen=self.smooth_window)
        self._Vq = deque(maxlen=self.smooth_window)

        self.t0 = time.time()
        plt.ion()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        self.ax_top, self.ax_mid = self.axs[0], self.axs[1]

        # --- A/V subplot
        (self.lA,) = self.ax_top.plot([], [], linewidth=1.3, label="Arousal (avg)")
        (self.lV,) = self.ax_top.plot([], [], linewidth=1.3, label="Valence (avg)")
        (self.lA_sm,) = self.ax_top.plot([], [], linestyle="--", linewidth=2.0, alpha=0.8, label="Arousal (sm)")
        (self.lV_sm,) = self.ax_top.plot([], [], linestyle="--", linewidth=2.0, alpha=0.8, label="Valence (sm)")

        if self.A_star is not None:
            self.ax_top.axhline(self.A_star, linestyle="--", linewidth=1.0, color="tab:green", alpha=0.6, label="A* target")
        if self.V_star is not None:
            self.ax_top.axhline(self.V_star, linestyle="--", linewidth=1.0, color="tab:red", alpha=0.6, label="V* target")

        self.ax_top.set_ylabel("Arousal / Valence")
        self.ax_top.set_ylim(0.0, 1.0)
        self.ax_top.grid(True, alpha=0.25)
        self.ax_top.legend(loc="upper right")

        # --- Params subplot
        self.lines_params = {
            "drums": self.ax_mid.plot([], [], linewidth=1.5, label="drums")[0],
            "pad":   self.ax_mid.plot([], [], linewidth=1.5, label="pad")[0],
            "tempo": self.ax_mid.plot([], [], linewidth=1.5, label="tempo")[0],
            "grain": self.ax_mid.plot([], [], linewidth=1.5, label="grain")[0],
        }
        self.ax_mid.set_ylabel("Music Params")
        self.ax_mid.set_ylim(0.0, 1.0)
        self.ax_mid.grid(True, alpha=0.25)
        self.ax_mid.legend(loc="upper right")
        self.ax_mid.set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show(block=False)

    def _smooth(self, seq, k):
        """Simple moving average of last k points (visual only)."""
        if k <= 1 or not seq:
            return list(seq)
        out, acc = [], deque(maxlen=k)
        for x in seq:
            acc.append(x)
            out.append(float(np.mean(acc)))
        return out

    def tick_epoch(self, t=None):
        """Mark an epoch boundary with a vertical dotted line on both axes."""
        if t is None:
            t = time.time() - self.t0
        for ax in (self.ax_top, self.ax_mid):
            ax.axvline(t, linestyle=":", linewidth=0.9, alpha=0.4)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, A_mean, V_mean, params_mean, **kwargs):
        """
        Add one averaged sample to the plot.

        Args:
            A_mean (float)
            V_mean (float)
            params_mean (dict): {'drums':..., 'pad':..., 'tempo':..., 'grain':...}
            **kwargs: ignored (keeps compatibility if caller passes extra args).
        """
        t = time.time() - self.t0
        self.times.append(t)
        self.A_vals.append(A_mean)
        self.V_vals.append(V_mean)

        for k in self.param_hist:
            self.param_hist[k].append(params_mean.get(k, np.nan))

        # Update A/V raw + smoothed
        self.lA.set_data(self.times, self.A_vals)
        self.lV.set_data(self.times, self.V_vals)
        self.lA_sm.set_data(self.times, self._smooth(self.A_vals, self.smooth_window))
        self.lV_sm.set_data(self.times, self._smooth(self.V_vals, self.smooth_window))

        # Update params
        for k, line in self.lines_params.items():
            line.set_data(self.times, self.param_hist[k])

        # Rescale and draw
        for ax in (self.ax_top, self.ax_mid):
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(0.0, 1.0)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
