# ---------- Global config ----------
import math

# FLAGS
FORCE_MOCK = True          # force mock even if Unicorn is available
ORACLE_WHEN_MOCK = True     # when we're in mock mode, use the A/V oracle instead of mock EEG

# DATA / WINDOWS
FS = 250
WIN_SEC = 2.0
HOP_SEC = 0.25
N_WIN = int(WIN_SEC * FS)
N_HOP = int(HOP_SEC * FS)

# OSC
OSC_IP, OSC_PORT = "127.0.0.1", 9000
ADDR_A = "/arousal"
ADDR_V = "/valence"
ADDR_ART = "/artifact"
OSC_PARAMS = ["drums", "pad", "tempo", "grain"]

# BANDS / CHANNELS
THETA = (4, 8)
ALPHA = (8, 12)
BETA  = (13, 30)
CH_NAMES = ["Fz","Cz","F3","F4","FP1","FP2","PO7","PO8"]

BASELINE_SEC = 20
FZ_SPIKE_UV = 1000.0

# EMA
EMA_TAU = 1.2
EMA_RATE = 1.0 / HOP_SEC
EMA_ALPHA = 1.0 - math.exp(-EMA_RATE / EMA_TAU)

# Targets
TARGETS = {
    "calm":     {"A_star": 0.30, "V_star": 0.80},
    "focus":    {"A_star": 0.45, "V_star": 0.55},
    "energize": {"A_star": 0.75, "V_star": 0.60},
}
ACTIVE_TARGET = "calm"
TARGET = TARGETS[ACTIVE_TARGET]

# Adaptive scaler
ADAPT_ALPHA = 0.01
K_SCALE = 2.0

# Optimizer defaults
DEFAULT_OPTIMIZER = "spsa"   # "spsa" or "bo"
