import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from training import Trainer
from networks import AutoEncoder
    
class Dataset:
    def __init__(self, data_csv: str, val_split: float = 0.1):
        df = pd.read_csv(data_csv)
        x_df = df.drop("y", axis=1)
        
        self.n_samples, self.n_features = x_df.shape
        data = np.ascontiguousarray(x_df.to_numpy(dtype=np.float32) / 255.0)
        self.data = data
        
        n_val = int(len(data) * val_split)
        indices = np.random.permutation(len(data))
        self.x_val = data[indices[:n_val]]
        self.x_train = data[indices[n_val:]]
        
        print(f"Loaded dataset with {self.n_samples} samples and {self.n_features} features.")


    
def main():
    dataset = Dataset('homework2/THA2train.csv', val_split=0.1)
    ae = AutoEncoder(act='relu', init='default')
    trainer = Trainer(model=ae, lr=0.01, batch_size=64, max_epochs=500)
    trainer.train(dataset.x_train, dataset.x_val)
    
if __name__ == '__main__':
    t0 = time.perf_counter()
    main()
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")