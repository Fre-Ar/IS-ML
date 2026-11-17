import numpy as np

def mse_metric(y_pred, y_true):
    """
    True pixel-wise Mean Squared Error in [0, 1] for reporting and plots.
    """
    return float(np.mean((y_pred - y_true) ** 2))

def mse_loss_and_grad(y_pred, y_true):
    # Mean over batch (N), sum over features (D)
    batch_size = y_true.shape[0]
    diff = y_pred - y_true
    loss = 0.5 * np.sum(diff ** 2) / batch_size
    grad = diff / batch_size
    return loss, grad

def bce_loss_and_grad(y_pred, y_true, eps = 1e-7):
    # Mean over batch (N), sum over features (D)
    batch_size = y_true.shape[0]
    # clipping y_pred to avoid log(y_pred) = NaN
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    loss = -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)) / batch_size
    
    grad = (y_pred - y_true) / (y_pred * (1.0 - y_pred) * batch_size)
    return loss, grad

loss_funcs = {
    "mse": mse_loss_and_grad,
    "bce": bce_loss_and_grad
}
    
    