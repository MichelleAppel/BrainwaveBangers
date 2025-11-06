# optimizers/bo.py
import numpy as np
from .base import BaseOptimizer

try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real
    HAVE_SKOPT = True
except Exception:
    HAVE_SKOPT = False


class BOOptimizer(BaseOptimizer):
    def __init__(
        self,
        bounds=None,
        seed=123,
        # exploration knobs (tuned)
        kappa0=2.5, kappa_min=1.0,
        eps0=0.20,  eps_min=0.03,
        decay=0.97,
        jitter=0.02,
        restart_every=80,
        history_warm=24
    ):
        self.bounds = bounds or [(0.0, 1.0)] * 4
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []
        self.best_x = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        self.best_y = -np.inf

        # exploration state
        self.kappa = float(kappa0); self.kappa_min = float(kappa_min)
        self.eps = float(eps0);     self.eps_min = float(eps_min)
        self.decay = float(decay)
        self.jitter = float(jitter)
        self.restart_every = int(restart_every)
        self.history_warm = int(history_warm)

        if HAVE_SKOPT:
            space = [Real(lo, hi, name=n)
                     for (lo, hi), n in zip(self.bounds, ["drums","pad","tempo","grain"])]
            self.opt = SkoptOptimizer(
                space,
                base_estimator="GP",
                acq_func="LCB",                        # Lower Confidence Bound (minimization)
                acq_func_kwargs={"kappa": self.kappa},
                noise="gaussian",
                random_state=seed
            )
        else:
            self.opt = None

        self._last_proposed = None
        self._iters = 0

    def name(self): return "bo"

    def start_epoch(self, current_distance: float):
        pass

    def _restart_gp(self):
        if not HAVE_SKOPT:
            return
        space = self.opt.space
        self.opt = SkoptOptimizer(
            space,
            base_estimator="GP",
            acq_func="LCB",
            acq_func_kwargs={"kappa": self.kappa},
            noise="gaussian",
            random_state=self.rng.integers(1, 2**31-1)
        )
        n = min(self.history_warm, len(self.X))
        for x, r in zip(self.X[-n:], self.y[-n:]):   # skopt minimizes
            self.opt.tell(x, -r)

    def propose(self, phase: int):
        if phase != 1:
            return None

        self._iters += 1
        # decay exploration
        self.kappa = max(self.kappa_min, self.kappa * self.decay)
        self.eps   = max(self.eps_min,   self.eps   * self.decay)

        if HAVE_SKOPT and (self._iters % self.restart_every == 0):
            self._restart_gp()

        # ε-greedy random step sometimes
        if self.rng.random() < self.eps:
            x = self.rng.uniform(0.0, 1.0, size=4)
        else:
            if HAVE_SKOPT:
                self.opt.acq_func_kwargs["kappa"] = float(self.kappa)
                x = np.array(self.opt.ask())
            else:
                sigma = max(0.08, 0.30 / np.sqrt(max(1, len(self.X))))
                x = np.clip(self.best_x + self.rng.normal(0, sigma, size=4), 0.0, 1.0)

        # trust region around current best (shrinks mildly)
        tr = 0.20 * max(0.3, 0.98 ** self._iters)
        x = np.clip(x, self.best_x - tr, self.best_x + tr)

        # tiny jitter
        if self.jitter > 0:
            x = np.clip(x + self.rng.normal(0, self.jitter, size=4), 0.0, 1.0)

        self._last_proposed = x
        return {"drums": float(x[0]), "pad": float(x[1]), "tempo": float(x[2]), "grain": float(x[3])}

    def report(self, phase: int, A_list, V_list):
        if not A_list or not V_list or self._last_proposed is None:
            return
        import math
        d = np.median([math.sqrt((a - self.A_star)**2 + (v - self.V_star)**2)
                       for a, v in zip(A_list, V_list)])
        reward = -float(d)

        x = self._last_proposed.copy()
        self.X.append(x.tolist()); self.y.append(reward)
        if reward > self.best_y:
            self.best_y = reward; self.best_x = x.copy()
        if HAVE_SKOPT:
            self.opt.tell(x.tolist(), -reward)  # skopt minimizes

    def step(self):
        pass

    def current_params_dict(self):
        x = self.best_x
        return {"drums": float(x[0]), "pad": float(x[1]), "tempo": float(x[2]), "grain": float(x[3])}
