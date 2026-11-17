import numpy as np
import numpy.random as random
from layers import Layer, LinearLayer

# === weights and bias initialization ===
def init_weights_zero(m: LinearLayer) -> None:
    """Zero init."""
    m.weight = np.zeros_like(m.weight)
    m.bias = np.zeros_like(m.bias)
        
def init_weights_uniform(m: LinearLayer, low: float = 0.0, high: float = 1.0) -> None:
    """Uniform init in [a, b)."""
    m.weight = random.uniform(low=low, high=high, size=m.weight.shape)
    m.bias = random.uniform(low=low, high=high, size=m.bias.shape)

def init_weights_normal(m: LinearLayer, mean: float = 0.0, std: float = 1.0) -> None:
    """Normal init with mean and std."""
    m.weight = random.normal(loc=mean, scale=std, size=m.weight.shape)
    m.bias = random.normal(loc=mean, scale=std, size=m.bias.shape)
        
def init_weights_default(m: LinearLayer) -> None:
    """Default pytorch init."""
    k = 1 / np.sqrt(m.weight.shape[0])
    init_weights_uniform(m, -k, k)
    
inits = {
    'zero': init_weights_zero,
    'default': init_weights_default,
    'uniform': init_weights_uniform,
    'normal': init_weights_normal
}
        
# Other candidates: Xavier init, Kaiming He init

def init_factory(type: str, m: Layer):
    if isinstance(m, LinearLayer) and hasattr(m, 'weight'):
        inits[type](m)

'''
def init_weights_smart(model: AutoEncoder, alpha: float = 0.01) -> None:
    """Applies He initialization before LeakyReLU/ReLU and Xavier before Tanh/Sigmoid."""
    for i in range(0, len(model.layers), 2):
        linear: LinearLayer = model.layers[i]
        act_layer = model.layers[i + 1]
        n_in, n_out = linear.weight.shape

        if isinstance(act_layer, (ReLU, LeakyReLU)):
            slope = alpha if isinstance(act_layer, LeakyReLU) else 0.0
            std = np.sqrt(2.0 / ((1.0 + slope ** 2) * n_in))
        else:  # Tanh, Sigmoid, Softplus
            std = np.sqrt(2.0 / (n_in + n_out))

        linear.weight = np.random.normal(0.0, std, size=(n_in, n_out)).astype(np.float32)
        linear.bias = np.zeros(n_out, dtype=np.float32)
'''