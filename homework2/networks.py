from layers import Layer, LinearLayer
from activations import activation_layers
from initialization import inits

class AutoEncoder(Layer):
    def __init__(self, layers = (784, 128, 16, 128, 784), act='relu', init='default'):
        super().__init__()
        self.layers: list[Layer] = []
        act_class = activation_layers[act]
        
        for i in range(len(layers) - 1):
            self.layers.append(act_class())
            linear = LinearLayer(layers[i], layers[i + 1])
            inits[init](linear)
            self.layers.append(linear)
        self.layers.append(act_class())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output