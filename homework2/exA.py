import numpy as np
import time
from training import Trainer
from networks import AutoEncoder
from data import Dataset
from plotting import plot_loss_curves, plot_reconstructions_per_category, plot_mse_per_category

SEED = 4

def main():
    t0 = time.perf_counter()
    
    np.random.seed(SEED)
    dataset = Dataset(train_csv='homework2/THA2train.csv', val_csv='homework2/THA2validate.csv')
    ae = AutoEncoder(acts=('leaky_relu', 'tanh', 'leaky_relu', 'sigmoid'), init='default')
    LR = 1e-2
    BATCH_SIZE = 64
    trainer = Trainer(model=ae, loss= 'bce', lr=LR, batch_size=BATCH_SIZE, max_epochs=500)
    trainer.train(dataset.train_x, dataset.val_x)
    
    dt = time.perf_counter() - t0
    print(f"Total Training time Elapsed: {dt:.3f}s")
    
    plot_loss_curves(trainer)
    plot_reconstructions_per_category(trainer.model, dataset.val_x, dataset.val_y)
    plot_mse_per_category(trainer.model, dataset.val_x, dataset.val_y)
    
if __name__ == '__main__':
    
    main()
    