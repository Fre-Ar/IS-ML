import numpy as np
import numpy.random as random
from layers import Layer, LinearLayer

# === weights and bias initialization ===
def init_zero(m: LinearLayer, **_) -> None:
    """Zero init."""
    m.weight.fill(0.0)
    m.bias.fill(0.0)
        
def init_uniform(m: LinearLayer, low: float = 0.0, high: float = 1.0, **_) -> None:
    """Uniform init in [a, b)."""
    m.weight[:] = random.uniform(low=low, high=high, size=m.weight.shape)
    m.bias[:] = random.uniform(low=low, high=high, size=m.bias.shape)

def init_normal(m: LinearLayer, mean: float = 0.0, std: float = 1.0, **_) -> None:
    """Normal init with mean and std."""
    m.weight[:] = random.normal(loc=mean, scale=std, size=m.weight.shape)
    m.bias[:] = random.normal(loc=mean, scale=std, size=m.bias.shape)
        
def init_default(m: LinearLayer, **_) -> None:
    """Default pytorch (LeCun uniform) init."""
    k = 1 / np.sqrt(m.weight.shape[0])
    init_uniform(m, low=-k, high=k)
    
def init_mixed(m: LinearLayer, act: str, alpha: float, **_) -> None:
    """Specialized init that inits biases at 0, but inits weights differently depending on the activation function applied after this linear layer.
    It applies He init before LeakyReLU/ReLU and Xavier init before Tanh/Sigmoid."""
    n_in, n_out = m.weight.shape
    if act in ['relu', 'leaky_relu']:
        # Kaiming/He normal init for ReLU variants
        slope = alpha if act=='leaky_relu' else 0.0
        std = np.sqrt(2.0 / ((1.0 + slope ** 2) * n_in))
    else: # Xavier normal init for Tanh, Sigmoid, Linear
        std = np.sqrt(2.0 / (n_in + n_out))
    mean = 0.0

    m.weight[:] = random.normal(loc=mean, scale=std, size=m.weight.shape)
    m.bias.fill(0.0)
    
INITS = {
    'zero': init_zero,
    'default': init_default,
    'uniform': init_uniform,
    'normal': init_normal,
    'mixed': init_mixed
}
        
# Other candidates: Xavier init, Kaiming He init

def init_factory(type: str, m: Layer, **kwargs):
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        INITS[type](m, **kwargs)
