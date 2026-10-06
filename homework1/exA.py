import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *


#1. data acquisition
df = pd.read_excel(INPUT_FILE_PATH, sheet_name="Sheet5")
start_index = (GROUP_NUMBER -1)*20+2
end_index = GROUP_NUMBER*20+1

data = df.iloc[start_index:end_index+1]
print(data)

#2. data transformation
#visualization of the data
plt.scatter(data['X'], data['Y'])
plt.title("Data Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()