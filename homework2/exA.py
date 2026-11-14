import numpy as np
import numpy.random as random
import time
import matplotlib.pyplot as plt
import pandas as pd
from activations import ReLU, Sigmoid, Tanh, Softplus

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
    
    
    
activation_layers = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softplus': Softplus
}

class AutoEncoder(Layer):
    def __init__(self, layers = (784, 128, 16, 128, 784), act='relu', init='default'):
        super().__init__()
        self.layers: list[LinearLayer] = []
        
        for i in range(len(layers) - 1):
            self.layers.append(LinearLayer(layers[i], layers[i+1]))
        self.activation = activation_layers[act]()
        
        # Initialize weights
        for layer in self.layers:
            inits[init](layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

def mse_loss_and_grad(y_pred, y_true):
    # Mean over batch (N), sum over features (D)
    batch_size = y_true.shape[0]
    diff = y_pred - y_true
    loss = 0.5 * np.sum(diff ** 2) / batch_size
    grad = diff / batch_size
    return loss, grad

class Dataset:
    def __init__(self, data_csv):
        self.data = pd.read_csv(data_csv)
        
class Trainer:
    """Trainer for the AutoEncoder.
    Args:
        model (AutoEncoder): The autoencoder model to be trained.
        lr (float): The learning rate for the optimizer.
        patience (int): The number of epochs to wait for improvement.
        min_delta (float): The minimum change in loss to qualify as improvement.
        max_epochs (int): The maximum number of epochs for training.
    """
    
    def __init__(self, 
        model: AutoEncoder,
        lr: float = 0.01,
        patience: int = 5,
        min_delta: float = 1e-5,
        batch_size: int = 32,
        max_epochs: int = 500):
        
        
        self.model = model
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.n_epochs = max_epochs
        
        self.history = []
        
    def train(self, x_train, epochs: int = 10):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.model.forward(x_train)
            loss, grad = mse_loss_and_grad(y_pred, x_train)
            self.history.append(loss)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')
            
            # Backward pass and weight update (not implemented)
            # This is a placeholder for the actual backpropagation logic.
            # In a real implementation, you would compute gradients and update weights here.
    
class SGD:
    def __init__(self, layers: list[Layer], lr: float = 0.01):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.weight -= self.lr * layer.grad_weight
            layer.bias -= self.lr * layer.grad_bias
    
def main():
    pass
    
if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")