import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import const.py from the utils package
from utils.const import *

#---------------------------------------------------------------------------------------
#A.I
#---------------------------------------------------------------------------------------
#1. data acquisition
df = pd.read_excel(INPUT_FILE_PATH, sheet_name="Sheet5")
print(df.head())
start_index = (GROUP_NUMBER -1)*20+2
end_index = GROUP_NUMBER*20+1

data = df.iloc[start_index:end_index+1]
print("Data acquired from the excel file:")
print(data)


#---------------------------------------------------------------------------------------

#2. data transformation
#visualization of the data as justification for min-max normalization (no big outliers)
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

# Extract the normalized X and Y values and row count for subsequent use
X_normalized = data['X_normalized'].values
Y_normalized = data['Y_normalized'].values
n = X_normalized.shape[0]

#---------------------------------------------------------------------------------------
#3. least squared closed form solution

# Add a bias term (intercept) to the input features (used later too)
X_with_bias = np.c_[np.ones(n), X_normalized]

def least_squared_closed_form(X_with_bias, Y_normalized):
    # Calculate the closed-form solution of least squares
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y_normalized

    return theta

ls_theta = least_squared_closed_form(X_with_bias, Y_normalized)
print(ls_theta)

# Sort data by X_normalized so lines plot smoothly without zig-zags
sorted_indices = np.argsort(data['X_normalized'].values)
x_sorted = data['X_normalized'].values[sorted_indices]
X_matrix = np.c_[np.ones(len(x_sorted)), x_sorted]

def plot_model(X_matrix, theta):
    """
    Plot the model predictions against the actual data.
    param X_matrix: Sorted input features with bias term (numpy array)
    param theta: Model parameters (numpy array)
    """
    data.plot(x='X_normalized', y='Y_normalized', kind='scatter')
    plt.plot(x_sorted, X_matrix @ theta, color='red')
    plt.show()
    
plot_model(X_matrix, ls_theta)

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
    pred_X = X @ theta
    # we use 1/2m to simplify the derivative of the cost for gradient descent.    
    cost = (1/(2*m)) * np.sum(np.square(Y - pred_X))
    return cost

#---------------------------------------------------------------------------------------
#5. Linear regression + gradient descent - gradient descent 1st iteration

def mse_gradient(X, Y, theta):
    """
    Calculate the gradient of the mean squared error cost function for linear regression.
    param X: Input features (numpy array)
    param Y: Target values (numpy array)
    param theta: Model parameters (numpy array)
    return: Gradient of the cost function (numpy array)
    """
    m = len(Y)
    pred_X = X @ theta
    gradient = -(1/m) * (X.T @ (Y - pred_X))
    return gradient

def gradient_descent(X, Y, theta, learning_rate, num_iterations, cost_function=mse_cost_function, cost_gradient=mse_gradient, save_costs=False):
    """
    Perform gradient descent to optimize the model parameters.
    param X: Input features (numpy array)
    param Y: Target values (numpy array)
    param theta: Initial model parameters (numpy array)
    param learning_rate: Learning rate for gradient descent (float)
    param num_iterations: Number of iterations to perform (int)
    cost_function: Function to compute the cost (callable)
    cost_gradient: Function to compute the gradient of the cost function (callable)
    save_costs: Whether to save the cost at each iteration (bool)
    return: Optimized model parameters (numpy array)
    """
    if save_costs:
        costs = [cost_function(X, Y, theta)]

    for _ in range(num_iterations):
        
        gradient = cost_gradient(X, Y, theta)
        theta = theta - learning_rate * gradient
        
        if save_costs:
            costs.append(cost_function(X, Y, theta))
            
    if save_costs:
        return theta, costs
    return theta

#first iteration of gradient descent
theta = np.array([0, 0])
learning_rate = 0.1

theta = gradient_descent(X_with_bias, Y_normalized, theta, learning_rate, 1)
print("Updated theta after one iteration of gradient descent:")
print(theta)

plot_model(X_matrix, theta)

#---------------------------------------------------------------------------------------
#6. Linear regression + gradient descent - 2nd iteration
theta = np.array([0, 0])
theta = gradient_descent(X_with_bias, Y_normalized, theta, learning_rate, 2)
print("Updated theta after two iterations of gradient descent:")
print(theta)

plot_model(X_matrix, theta)

#---------------------------------------------------------------------------------------
#7. Linear regression + gradient descent - n iterations
theta = np.array([0, 0])
iterations = 500
theta, costs = gradient_descent(X_with_bias, Y_normalized, theta, learning_rate, iterations, save_costs=True)

# Plot the costs over iterations to visualize convergence
plt.plot(costs)
plt.title(f"Costs over {iterations} iterations")
plt.xlabel("Iteration number")
plt.ylabel("MSE Cost value")
plt.show()

print(f"Final cost: {costs[-1]}")

# plot the final model after n iterations
print(f"Updated theta after {iterations} iterations of gradient descent:")
print(theta)
plot_model(X_matrix, theta)


#---------------------------------------------------------------------------------------
#A.II.
#---------------------------------------------------------------------------------------

#polynomial regression

def cost_function(X, Y, theta):
    n = len(Y)
    pred_X = X @ theta
    cost = (1/(4*n)) * np.sum(np.power(Y - pred_X,4))
    return cost

def cost_gradient(X, Y, theta):
    n = len(Y)
    pred_X = X @ theta
    gradient = -(1/n) * (X.T @ np.power((Y - pred_X),3))
    return gradient

theta = np.array([0, 0, 0])
learning_rate = 0.1
iterations = 1000

X = np.column_stack([np.ones(n), X_normalized, X_normalized**2])
theta, costs = gradient_descent(X, Y_normalized, theta, learning_rate, iterations, cost_function=cost_function, cost_gradient=cost_gradient, save_costs=True)

# Plot the costs over iterations to visualize convergence
plt.plot(costs)
plt.title(f"Costs over {iterations} iterations")
plt.xlabel("Iteration number")
plt.ylabel("Cost value")
plt.show()

print(f"Final cost: {costs[-1]}")

# plot the final model after n iterations
print(f"Updated theta after {iterations} iterations of gradient descent:")
print(theta)

X_matrix = np.c_[np.ones(len(x_sorted)), x_sorted, x_sorted**2]
plot_model(X_matrix, theta)