import numpy as np
from .base import BaseOptimizer

# We try to use skopt if available. Otherwise: fallback to a strong random-search + annealing.
try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real
    HAVE_SKOPT = True
except Exception:
    HAVE_SKOPT = False

class BOOptimizer(BaseOptimizer):
    def __init__(self, bounds=None, seed=123):
        self.bounds = bounds or [(0.0,1.0)]*4
        self.rng = np.random.default_rng(seed)
        self.X, self.y = [], []  # history
        self.best_x = np.array([0.5,0.5,0.5,0.5], dtype=float)
        self.best_y = -np.inf
        if HAVE_SKOPT:
            space = [Real(lo, hi, name=n) for (lo,hi), n in zip(self.bounds, ["drums","pad","tempo","grain"])]
            self.opt = SkoptOptimizer(
                space, base_estimator="GP", acq_func="EI",
                acq_func_kwargs={"xi": 0.05},   # was implicit default ~0.01–0.1
                random_state=seed, noise="gaussian"
            )
        else:
            self.opt = None
        self._last_proposed = None

    def name(self): return "bo"

    def start_epoch(self, current_distance: float):
        # BO is single-phase (we evaluate one candidate per epoch).
        pass

    def propose(self, phase: int):
        if phase != 1:
            return None
        if HAVE_SKOPT:
            x = np.array(self.opt.ask())
        else:
            if len(self.X) < 10:
                x = self.rng.uniform(0.0,1.0,size=4)
            else:
                # anneal around best
                sigma = max(0.05, 0.25 / np.sqrt(len(self.X)))
                x = np.clip(self.best_x + self.rng.normal(0, sigma, size=4), 0.0, 1.0)
        self._last_proposed = x
        return {"drums": float(x[0]), "pad": float(x[1]), "tempo": float(x[2]), "grain": float(x[3])}

    def report(self, phase: int, A_list, V_list):
        # Convert collected samples to a scalar reward = -mean distance
        if not A_list or not V_list or self._last_proposed is None:
            return
        import math
        d = np.mean([math.sqrt((a- self.A_star)**2 + (v- self.V_star)**2) for a,v in zip(A_list, V_list)])
        reward = -float(d)
        x = self._last_proposed.copy()
        self.X.append(x.tolist()); self.y.append(reward)
        if reward > self.best_y:
            self.best_y = reward; self.best_x = x.copy()
        if HAVE_SKOPT:
            self.opt.tell(x.tolist(), -reward)  # skopt minimizes

    def step(self):
        # nothing else to do; current center is best observed so far
        pass

    def current_params_dict(self):
        x = self.best_x
        return {"drums": float(x[0]), "pad": float(x[1]), "tempo": float(x[2]), "grain": float(x[3])}
