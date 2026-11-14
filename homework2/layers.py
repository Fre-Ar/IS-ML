import numpy as np
import numpy.random as random

class Layer:
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError
    def __call__(self, x): 
        return self.forward(x)

class LinearLayer(Layer):
    # ax+b
    def __init__(self, input_dim, output_dim):
        k = 1/np.sqrt(input_dim)
        self.weight = random.uniform(low=-k, high=k, size=(input_dim, output_dim))
        self.bias = random.uniform(low=-k, high=k, size=output_dim)
        
        self.x = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x):
        self.x = x
        return x @ self.weight + self.bias
    
    def backward(self, grad_output):
        self.grad_weight = self.x.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weight.T
    
    