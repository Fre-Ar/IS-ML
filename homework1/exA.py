import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *

#---------------------------------------------------------------------------------------
#A.I.
#---------------------------------------------------------------------------------------
#1. data acquisition
df = pd.read_excel(INPUT_FILE_PATH, sheet_name="Sheet5")
start_index = (GROUP_NUMBER -1)*20+2
end_index = GROUP_NUMBER*20+1

data = df.iloc[start_index:end_index+1]
print("Data acquired from the excel file:")
print(data)


#---------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------
#3. least squared closed form solution

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

#---------------------------------------------------------------------------------------
#4.Linear regression + gradient descent - cost function
# we will use mean squared error as the cost function (penalizing large errors more than small errors)

def mse_cost_function(X, Y, theta):
    """
    Calculate the mean squared error cost function for linear regression.
    param X: Input features (numpy array)
    param Y: Target values (numpy array)
    param theta: Model parameters (numpy array)
    return: Mean squared error cost (float)
    """
    m = len(Y)
    predictions = X @ theta
    #we use 1/2m to simplify the derivative of the cost for gradient descent.    
    cost = (1/(2*m)) * np.sum(np.square(predictions - Y))
    return cost

#---------------------------------------------------------------------------------------
#5. Linear regression + gradient descent - gradient descent 1st iteration

def gradient_mse(X,Y,theta):
    """
    Calculate the gradient of the mean squared error cost function for linear regression.
    param X: Input features (numpy array)
    param Y: Target values (numpy array)
    param theta: Model parameters (numpy array)
    return: Gradient of the cost function (numpy array)
    """
    m = len(Y)
    predictions = X @ theta
    gradient = (1/m) * (X.T @ (predictions - Y))
    return gradient

#first iteration of gradient descent
theta = np.array([0, 0])
learning_rate = 0.01
X_with_bias = np.c_[np.ones(data['X_normalized'].shape[0]), data['X_normalized']]
Y_normalized = data['Y_normalized'].values
gradient = gradient_mse(X_with_bias, Y_normalized, theta)
theta = theta - learning_rate * gradient
print("Updated theta after one iteration of gradient descent:")
print(theta)

#---------------------------------------------------------------------------------------
#6. Linear regression + gradient descent - 2nd iteration
gradient = gradient_mse(X_with_bias, Y_normalized, theta)
theta = theta - learning_rate * gradient
print("Updated theta after two iterations of gradient descent:")
print(theta)
#TODO:plotting

#---------------------------------------------------------------------------------------
#7. Linear regression + gradient descent - n iterations
def gradient_descent(X, Y, theta, learning_rate, num_iterations):
    """
    Perform gradient descent to optimize the model parameters.
    param X: Input features (numpy array)
    param Y: Target values (numpy array)
    param theta: Initial model parameters (numpy array)
    param learning_rate: Learning rate for gradient descent (float)
    param num_iterations: Number of iterations to perform (int)
    return: Optimized model parameters (numpy array)
    """
    for i in range(num_iterations):
        gradient = gradient_mse(X, Y, theta)
        theta = theta - learning_rate * gradient
    return theta
X= data['X_normalized'].values
Y= data['Y_normalized'].values
theta = gradient_descent(X_with_bias, Y_normalized, theta, learning_rate, 1000)

Y_pred = X_with_bias @ theta

#TODO:plotting

#---------------------------------------------------------------------------------------
#A.II.
#---------------------------------------------------------------------------------------

#polynomial regression

def regression_model(x,theta):
    """
    Calculate the predictions of the regression model.
    param x: Input features (numpy array)
    param theta: Model parameters (numpy array)
    return: Predictions (numpy array)
    """
    # Construct X = [1, x, x^2] inside the function
    x = x.reshape(-1, 1)  # Ensure X is a column vector
    X = np.hstack([np.ones_like(x), x, x**2])
    
    return X @ theta
