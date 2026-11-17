import copy
import numpy as np
from layers import Layer, LinearLayer
from networks import AutoEncoder
from metrics import loss_funcs, mse_metric
 
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
    Handles mini-batch SGD training, evaluation, and checkpointing for an AutoEncoder.

    Manages the training loop over mini-batches, logs training and validation losses alongside standardized pixel-wise MSE, detects numerical divergence (NaN/Inf), supports early stopping with best-weight restoration, and optionally captures weight snapshots at initialization, mid-training, and completion.
    
    Args:
        model (AutoEncoder): The autoencoder model to be trained.
        loss (str): Name of loss function used for training, "mse" or "bce".
        lr (float): Learning rate for the SGD optimizer. Defaults to 0.01.
        early_stopping (bool): Whether to stop early when validation loss stops improving and restore the best weights. Defaults to True.
        patience (int): Number of epochs to wait for validation improvement before triggering early stopping. Defaults to 5.
        min_delta (float): Mminimum decrease in validation loss to qualify as improvement. Defaults to 1e-5.
        batch_size (int): Number of samples per mini-batch. Defaults to 64.
        max_epochs (int): Maximum number of training epochs. Defaults to 500.
        capture_snapshots (bool): If True, saves weight snapshots at initialization, halfway through, and at the end of training. Defaults to False.
        verbose (bool): Whether to print epoch-by-epoch training logs. Defaults to True.
        
    Attributes:
        optimizer (SGD): SGD optimizer updating linear layers.
        history (dict[str, list[float]]): Dictionary of the eecorded per-epoch metrics with keys
            'train_loss', 'val_loss', and 'val_mse'.
        snapshots (dict): Stored weight and bias snapshots when capture_snapshots is True. Can be restored into model with `_restore_weights_snapshot(snapshots)` 
        has_diverged (bool): Set to True if training terminates early due to NaN or infinite loss values.
    """
    
    def __init__(self, 
        model: AutoEncoder,
        loss: str,
        lr: float = 0.01,
        early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 1e-6,
        batch_size: int = 64,
        max_epochs: int = 500,
        capture_snapshots: bool = False,
        verbose: str = True):
        
        self.model = model
        self.optimizer = SGD(model.layers, lr=lr)
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        
        self.loss_str = loss
        self.loss_func = loss_funcs[loss]
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_mse': []}
        
        self.capture_snapshots = capture_snapshots
        self.snapshots = {}
        
        self.verbose = verbose
        self.has_diverged = False
        
    def _iterate_batches(self, x: np.ndarray, shuffle: bool = True):
        n_samples = x.shape[0]
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n_samples, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield x[batch_idx]
            
    def _evaluate(self, x: np.ndarray):
        # evaluate using both the chosen mse regardless of training loss, as per assignment
        total_loss = 0.0
        total_mse = 0.0
        n_samples = x.shape[0]
        
        for x_batch in self._iterate_batches(x, shuffle=False):
            y_pred = self.model.forward(x_batch)
            loss, _ = self.loss_func(y_pred, x_batch)
            batch_mse = mse_metric(y_pred, x_batch)
            
            # weight by batch size so partial final batches don't skew the mean
            total_loss += loss * x_batch.shape[0]
            total_mse += batch_mse * x_batch.shape[0]
            
        return total_loss / n_samples, total_mse / n_samples
    
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
        
        if self.capture_snapshots:
            self.snapshots['init'] = self._get_weights_snapshot()
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0

            # 1. Mini-batch training pass
            for x_batch in self._iterate_batches(x_train, shuffle=True):
                y_pred = self.model.forward(x_batch)
                loss, grad_out = self.loss_func(y_pred, x_batch)

                self.model.backward(grad_out)
                self.optimizer.step()

                epoch_loss += loss * x_batch.shape[0]

            train_loss = epoch_loss / n_samples
            self.history['train_loss'].append(train_loss)
            
            # 1.a. Check for NaN / Inf divergence
            if not np.isfinite(train_loss):
                self.has_diverged = True
                if self.verbose:
                    print(f"Training diverged (NaN/Inf) at epoch {epoch + 1}.")
                break
            
            # 2. Validation pass
            val_loss, val_mse = self._evaluate(x_val)

            self.history['val_loss'].append(val_loss)
            self.history['val_mse'].append(val_mse)
            
            log_str = (
                f"Epoch {epoch + 1}/{self.max_epochs} | "
                f"Train {self.loss_str.upper()}: {train_loss:.5f} | "
                f"Val {self.loss_str.upper()}: {val_loss:.5f} | "
                f"Val MSE: {val_mse:.5f}"
            )

            if self.verbose:
                print(log_str)
                
            # Capture halfway snapshot 
            if self.capture_snapshots and (epoch + 1) == (self.max_epochs // 2):
                self.snapshots['half'] = self._get_weights_snapshot()
            
            # 3. Early stopping check
            if self.early_stopping:
                if best_loss - val_loss > self.min_delta:
                    best_loss = val_loss
                    best_weights = self._get_weights_snapshot()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best weights (loss: {best_loss:.5f}).")
                        if best_weights is not None:
                            self._restore_weights_snapshot(best_weights)
                        break
                    
        if self.capture_snapshots:
            self.snapshots['end'] = self._get_weights_snapshot()
