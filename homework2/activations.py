import numpy as np
from exA import Layer

def clip(x):
    return np.clip(x, -1028, 1028)  

class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask

class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        # Prevent overflow
        x = clip(x)
        self.out = 1 / (1 + np.exp(-x)) 
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1 - self.out)


class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1 - self.out ** 2)


class Softplus(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.logaddexp(0, x)

    def backward(self, grad_output):
        x = clip(self.x)
        sig = 1 / (1 + np.exp(-x))
        return grad_output * sig