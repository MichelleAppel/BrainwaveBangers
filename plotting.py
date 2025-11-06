import time
import numpy as np
import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self, A_star=None, V_star=None):
        self.times = []
        self.A_means = []
        self.V_means = []
        self.param_means = {k: [] for k in ["drums", "pad", "tempo", "grain"]}
        self.t0 = time.time()

        plt.ion()
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # A/V subplot
        self.lA, = self.axs[0].plot([], [], "C0-o", label="Arousal (avg)")
        self.lV, = self.axs[0].plot([], [], "C1-o", label="Valence (avg)")
        if A_star is not None:
            self.axs[0].axhline(A_star, linestyle="--", linewidth=1, color="C2", label="A* target")
        if V_star is not None:
            self.axs[0].axhline(V_star, linestyle="--", linewidth=1, color="C3", label="V* target")
        self.axs[0].set_ylabel("Arousal / Valence")
        self.axs[0].legend(loc="upper right")

        # Params subplot
        self.lines_params = {k: self.axs[1].plot([], [], "-o", label=k)[0]
                             for k in self.param_means}
        self.axs[1].legend(loc="upper right")
        self.axs[1].set_ylabel("Music Params")
        self.axs[1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show(block=False)

    def update(self, A_mean, V_mean, params_mean):
        t = time.time() - self.t0
        self.times.append(t)
        self.A_means.append(A_mean)
        self.V_means.append(V_mean)
        for k in self.param_means:
            self.param_means[k].append(params_mean.get(k, np.nan))

        self.lA.set_data(self.times, self.A_means)
        self.lV.set_data(self.times, self.V_means)
        for k, line in self.lines_params.items():
            line.set_data(self.times, self.param_means[k])

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
