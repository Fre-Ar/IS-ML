import numpy as np

def mse_loss_and_grad(y_pred, y_true):
    # Mean over batch (N), sum over features (D)
    batch_size = y_true.shape[0]
    diff = y_pred - y_true
    loss = 0.5 * np.sum(diff ** 2) / batch_size
    grad = diff / batch_size
    return loss, grad