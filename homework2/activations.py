import numpy as np
from layers import Layer

def clip(x):
    return np.clip(x, -1028.0, 1028.0)  

class ReLU(Layer):
    def __init__(self, **_):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask
    
class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.mask = None
        self.alpha = alpha

    def forward(self, x):
        self.mask = np.where(x > 0, 1.0, self.alpha)
        return np.maximum(self.alpha * x, x)

    def backward(self, grad_output):
        return grad_output * self.mask
    
class ELU(Layer):
    def __init__(self, alpha=0.1):
        self.mask = None
        self.alpha = alpha

    def forward(self, x):
        out = np.where(x > 0, x, self.alpha * np.expm1(x))
        # reusing out to avoid a second exp call
        self.mask = np.where(x > 0, 1.0, out + self.alpha)
        return out

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid(Layer):
    def __init__(self, **_):
        self.out = None

    def forward(self, x):
        # Prevent overflow
        x = clip(x)
        self.out = 1 / (1 + np.exp(-x)) 
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Tanh(Layer):
    def __init__(self, **_):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)


class Softplus(Layer):
    def __init__(self, **_):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.logaddexp(0, x)

    def backward(self, grad_output):
        x = clip(self.x)
        sig = 1 / (1 + np.exp(-x))
        return grad_output * sig
    
class Linear(Layer):
    def __init__(self, **_):
        pass
    
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output 
    
    
activation_layers = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'elu': ELU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softplus': Softplus,
    'linear': Linear
}