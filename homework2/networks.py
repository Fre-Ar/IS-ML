from layers import Layer, LinearLayer
from activations import activation_layers
from initialization import INITS

class AutoEncoder(Layer):
    def __init__(self,
                 layers = (784, 128, 16, 128, 784),
                 acts: tuple[str,...] = activation_layers.keys(),
                 init: str = 'default',
                 alpha = 0.01):
        super().__init__()
        n_layers = len(layers) - 1
        assert len(acts) == n_layers, "Must provide one act per layer transition."
        self.layers: list[Layer] = []
        
        for i in range(len(layers) - 1):
            linear = LinearLayer(layers[i], layers[i + 1])
            INITS[init](linear, act=acts[i], alpha=alpha)
            self.layers.append(linear)
            self.layers.append(activation_layers[acts[i]](alpha=alpha))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def forward_with_states(self, x):
        """Returns output plus post-activation outputs [h0, z, h1, x_hat] for Part B.3."""
        states = []
        for layer in self.layers:
            x = layer(x)
            if not isinstance(layer, LinearLayer):
                states.append(x)
        return x, states