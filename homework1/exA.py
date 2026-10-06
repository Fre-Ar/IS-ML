import sys
import pandas as pd
import numpy as np

# Import const.py from the utils package
from utils import const

df = pd.read_excel(const.INPUT_FILE_PATH, sheet_name="Sheet5")
print(df.head())