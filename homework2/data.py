import pandas as pd
import numpy as np

def load_csv(csv):
    df = pd.read_csv(csv)
    y = df["y"].to_numpy(dtype=np.int8)
    x_df = df.drop("y", axis=1)
    x = np.ascontiguousarray(x_df.to_numpy(dtype=np.float32) / 255.0)
    
    return x, y

class Dataset:
    def __init__(self, train_csv: str, val_csv: str):
        train_x, train_y= load_csv(train_csv)
        val_x, val_y = load_csv(val_csv)
        
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
       
# Fashion-MNIST class names, the index of the list is the corresponding y value
CLASS_NAMES = [
    "0: T-shirt/top",
    "1: Trouser",
    "2: Pullover",
    "3: Dress",
    "4: Coat",
    "5: Sandal",
    "6: Shirt",
    "7: Sneaker",
    "8: Bag",
    "9: Ankle boot",
]
