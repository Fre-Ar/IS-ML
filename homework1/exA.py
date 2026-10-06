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
print("Data acquired from the excel file:")
print(data)

#2. data transformation
#visualization of the data as justification for min-max normalization
data.plot(x='X', y='Y', kind='scatter')
plt.show()
#min max normalisation
minX = data['X'].min()
maxX = data['X'].max()
minY = data['Y'].min()
maxY = data['Y'].max()
# Normalize the data
data['X_normalized'] = (data['X'] - minX) / (maxX - minX)
data['Y_normalized'] = (data['Y'] - minY) / (maxY - minY)

print("Data after min-max normalization:")
print(data)

def least_squared_closed_form(data):
    # Extract the normalized X and Y values
    X_normalized = data['X_normalized'].values
    Y_normalized = data['Y_normalized'].values

    # Add a bias term (intercept) to the input features
    X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

    # Calculate the closed-form solution for linear regression
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y_normalized

    return theta

ls_result = least_squared_closed_form(data)
print(ls_result)

print("Data after min-max normalization:")
print(data)

data.plot(x='X_normalized', y='Y_normalized', kind='scatter')
plt.plot(data['X_normalized'], ls_result[0] + ls_result[1] * data['X_normalized'], color='red')
plt.show()