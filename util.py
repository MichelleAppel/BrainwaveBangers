import math
from config import TARGET, K_SCALE

def ema(prev, x, alpha):
    if prev is None:
        return x
    return (1 - alpha) * prev + alpha * x

def distance(A, V):
    return math.sqrt((A - TARGET["A_star"])**2 + (V - TARGET["V_star"])**2)

def squash_tanh(x, m, s, k=K_SCALE):
    z = (x - m) / (k * s + 1e-9)
    return 0.5 + 0.5 * math.tanh(z)
