import math, numpy as np
from .base import BaseOptimizer

class SPSAOptimizer(BaseOptimizer):
    def __init__(self, dim=4, low=0.0, high=1.0,
                 alpha0=0.15, c0=0.12, momentum=0.8,
                 sigma0=0.03, sigma_decay=0.995,
                 min_c=0.03, stuck_eps=1e-3, stuck_kick=0.08,
                 seed=123):
        self.dim = dim
        self.low, self.high = low, high
        self.alpha0 = alpha0
        self.c0 = c0
        self.momentum = momentum
        self.sigma = sigma0
        self.sigma_decay = sigma_decay
        self.min_c = min_c
        self.stuck_eps = stuck_eps
        self.stuck_kick = stuck_kick
        self.rng = np.random.default_rng(seed)
        self.p = np.full(dim, 0.5)
        self.v = np.zeros(dim)
        self._Delta = None
        self._r_plus = None
        self._r_minus = None
        self._prev_distance = None
        self.t = 0

    def name(self): return "spsa"

    def _proj(self, x): return np.clip(x, self.low, self.high)
    def _alpha(self): return self.alpha0 / (1.0 + 0.02 * self.t)
    def _c(self):     return max(self.min_c, self.c0 / math.sqrt(1.0 + 0.01 * self.t))

    def start_epoch(self, current_distance: float):
        self.t += 1
        self._prev_distance = current_distance
        self._Delta = self.rng.choice([-1.0, 1.0], size=self.dim)
        self.p = self._proj(self.p + self.rng.normal(0.0, self.sigma, size=self.dim))
        self.sigma = max(0.005, self.sigma * self.sigma_decay)

    def propose(self, phase: int):
        c = self._c()
        if phase == 1:
            x = self._proj(self.p + c * self._Delta)
        elif phase == 2:
            x = self._proj(self.p - c * self._Delta)
        else:
            return None
        return {"drums": float(x[0]), "pad": float(x[1]), "tempo": float(x[2]), "grain": float(x[3])}

    def report(self, phase: int, A_list, V_list):
        # Use averaged distance over the phase window
        if len(A_list) and len(V_list):
            import numpy as np
            from math import sqrt
            # We'll pass the actual distances from main; here we convert to a scalar
            d = np.mean([sqrt((a - self.A_star)**2 + (v - self.V_star)**2) for a, v in zip(A_list, V_list)]) \
                if hasattr(self, "A_star") else np.nan
        # But we keep original behavior: distances are computed in main and passed via observe_* below.
        pass

    # For API compatibility with main, we expose observe_* and update
    def observe_reward_plus(self, new_distance):
        if self._prev_distance is None:
            self._r_plus = 0.0
        else:
            self._r_plus = float(np.clip(self._prev_distance - new_distance, -0.4, 0.4))

    def observe_reward_minus(self, new_distance):
        if self._prev_distance is None:
            self._r_minus = 0.0
        else:
            self._r_minus = float(np.clip(self._prev_distance - new_distance, -0.4, 0.4))

    def step(self):
        # perform parameter update at end of epoch
        if self._r_plus is None or self._r_minus is None or self._Delta is None:
            return
        c = self._c()
        g_hat = ((self._r_plus - self._r_minus) / (2.0 * c)) * self._Delta
        self.v = self.momentum * self.v + (1.0 - self.momentum) * g_hat
        self.p = self._proj(self.p + self._alpha() * self.v)
        if abs(self._r_plus - self._r_minus) < self.stuck_eps:
            kick = self.rng.normal(0.0, self.stuck_kick, size=self.dim)
            self.p = self._proj(self.p + kick)
        self._Delta = None
        self._r_plus = None
        self._r_minus = None
        self._prev_distance = None

    def current_params_dict(self):
        return {"drums": float(self.p[0]), "pad": float(self.p[1]),
                "tempo": float(self.p[2]), "grain": float(self.p[3])}
