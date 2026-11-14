import numpy as np
import numpy.random as random
from layers import LinearLayer

# === weights and bias initialization ===
# TODO: fix bias init
def init_weights_zero(m) -> None:
    """Zero init."""
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        m.weight = np.zeros_like(m.weight)
        m.bias = np.zeros_like(m.bias)
        
def init_weights_default(m) -> None:
    """Default init."""
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        k = 1 / np.sqrt(m.weight.shape[0])
        m.weight = random.uniform(low=-k, high=k, size=m.weight.shape)
        m.bias = random.uniform(low=-k, high=k, size=m.bias.shape)

def init_weights_uniform(m, low: float = 0.0, high: float = 1.0) -> None:
    """Uniform init in [a, b)."""
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        m.weight = random.uniform(low=low, high=high, size=m.weight.shape)

def init_weights_normal(m, mean: float = 0.0, std: float = 1.0) -> None:
    """Normal init with mean and std."""
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        m.weight = random.normal(loc=mean, scale=std, size=m.weight.shape)

inits = {
    'zero': init_weights_zero,
    'default': init_weights_default,
    'uniform': init_weights_uniform,
    'normal': init_weights_normal
}