import copy
import numpy as np
import numpy.random as random
from layers import Layer, LinearLayer
from networks import AutoEncoder
from metrics import mse_loss_and_grad
 
class SGD:
    def __init__(self, layers: list[Layer], lr: float = 0.01):
        self.layers = [l for l in layers if isinstance(l, LinearLayer)]
        self.lr = lr

    def step(self):
        for layer in self.layers:
            layer.weight -= self.lr * layer.grad_weight
            layer.bias -= self.lr * layer.grad_bias 

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
        batch_size: int = 64,
        max_epochs: int = 500):
        
        self.model = model
        self.optimizer = SGD(model.layers, lr=lr)
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        self.history = {'train_loss': [], 'val_loss': []}
        
    def _iterate_batches(self, x: np.ndarray, shuffle: bool = True):
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield x[batch_idx]
            
    def _evaluate(self, x: np.ndarray) -> float:
        total_loss = 0.0
        n_samples = x.shape[0]
        for x_batch in self._iterate_batches(x, shuffle=False):
            y_pred = self.model.forward(x_batch)
            loss, _ = mse_loss_and_grad(y_pred, x_batch)
            # weight by batch size so partial final batches don't skew the mean
            total_loss += loss * x_batch.shape[0]
        return total_loss / n_samples
    
    def _get_weights_snapshot(self):
        return [
            (copy.deepcopy(layer.weight), copy.deepcopy(layer.bias))
            for layer in self.model.layers
            if isinstance(layer, LinearLayer)
        ]

    def _restore_weights_snapshot(self, snapshot):
        linear_layers = [l for l in self.model.layers if isinstance(l, LinearLayer)]
        for layer, (w, b) in zip(linear_layers, snapshot):
            layer.weight = w
            layer.bias = b
        
    def train(self, x_train, x_val):
        best_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0
        n_samples = x_train.shape[0]
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0

            # 1. Mini-batch training pass
            for x_batch in self._iterate_batches(x_train, shuffle=True):
                y_pred = self.model.forward(x_batch)
                loss, grad_out = mse_loss_and_grad(y_pred, x_batch)

                self.model.backward(grad_out)
                self.optimizer.step()

                epoch_loss += loss * x_batch.shape[0]

            train_loss = epoch_loss / n_samples
            self.history['train_loss'].append(train_loss)
            
            # 2. Validation pass (fallback to train_loss if no val set given)
            if x_val is not None:
                val_loss = self._evaluate(x_val)
                self.history['val_loss'].append(val_loss)
                monitored_loss = val_loss
                log_str = f"Epoch {epoch + 1}/{self.max_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}"
            else:
                monitored_loss = train_loss
                log_str = f"Epoch {epoch + 1}/{self.max_epochs} | Train Loss: {train_loss:.5f}"

            print(log_str)
            
            # 3. Early stopping check
            if best_loss - monitored_loss > self.min_delta:
                best_loss = monitored_loss
                best_weights = self._get_weights_snapshot()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights (loss: {best_loss:.5f}).")
                    if best_weights is not None:
                        self._restore_weights_snapshot(best_weights)
                    break
