from .spsa import SPSAOptimizer
from .bo import BOOptimizer

def get_optimizer(name: str):
    name = (name or "spsa").lower()
    if name == "spsa":
        return SPSAOptimizer()
    elif name == "bo":
        opt = BOOptimizer()
        # pass targets later from main (A*, V*) so BO can compute reward
        return opt
    else:
        raise ValueError(f"Unknown optimizer: {name}")
