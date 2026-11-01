import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.const import *

#---------------------------------------------------------------------------------------
# A.I
#---------------------------------------------------------------------------------------
# 1. data acquisition
df = pd.read_excel(INPUT_FILE_PATH, sheet_name="Sheet5")
print(df.head())
start_index = (GROUP_NUMBER -1)*20+2
end_index = GROUP_NUMBER*20+1

data = df.iloc[start_index:end_index+1]
print("Data acquired from the excel file:")
print(data)


#---------------------------------------------------------------------------------------

# 2. data transformation (Z-score standardization)
# visualization of the data 
data.plot(x='X', y='Y', kind='scatter', title='Scatter plot of the data (pre-transformation)')
plt.show()
# z-score standardization
meanX, stdX = data['X'].mean(), data['X'].std()
meanY, stdY = data['Y'].mean(), data['Y'].std()
data['X_standardized'] = (data['X'] - meanX) / stdX
data['Y_standardized'] = (data['Y'] - meanY) / stdY

print("Data after Z-score standardization:")
print(data)
data.plot(x='X_standardized', y='Y_standardized', kind='scatter', title='Scatter plot of the data (post-standardization)')
plt.show()

# Extract the standardized X and Y values and row count for subsequent use
X_s = data['X_standardized'].values
Y_s = data['Y_standardized'].values
n = X_s.shape[0]

#---------------------------------------------------------------------------------------
# 3. least squares closed form solution

# Add a bias term (intercept) to the input features (used later too)
# so multiplication with theta will include the intercept term theta_0
X_with_bias = np.c_[np.ones(n), X_s]

def least_squared_closed_form(X_with_bias, Y_s):
    # Calculate the closed-form solution of least squares
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ Y_s

    return theta

ls_theta = least_squared_closed_form(X_with_bias, Y_s)
print(ls_theta)

# Sort data by X_s so lines plot smoothly without zig-zags
sorted_indices = np.argsort(data['X_standardized'].values)
x_sorted = data['X_standardized'].values[sorted_indices]
X_matrix = np.c_[np.ones(len(x_sorted)), x_sorted]

def plot_model(X_matrix, theta, title=""):
    """
    Plot the model predictions against the actual data.
    param X_matrix: Sorted input features with bias term (numpy array)
    param theta: Model parameters (numpy array)
    param title: Title for the plot
    """
    data.plot(x='X_standardized', y='Y_standardized', kind='scatter', title=title)
    plt.plot(x_sorted, X_matrix @ theta, color='red')
    plt.show()
    
plot_model(X_matrix, ls_theta, title="Least Squares Closed Form Solution")

#---------------------------------------------------------------------------------------
# 4.Linear regression + gradient descent - cost function
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
# 5. Linear regression + gradient descent - gradient descent 1st iteration

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
    gradient = (1/m) * (X.T @ (pred_X - Y))
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
rng = np.random.default_rng(SEED)
starting_theta = rng.standard_normal(2)
print(f"Starting theta for linear regression: {starting_theta}")
learning_rate = 0.1

theta = starting_theta.copy()
theta = gradient_descent(X_with_bias, Y_s, theta, learning_rate, 1)
print("Updated theta after one iteration of gradient descent:")
print(theta)

plot_model(X_matrix, theta, title="After 1 Iteration")

#---------------------------------------------------------------------------------------
# 6. Linear regression + gradient descent - 2nd iteration
theta = starting_theta.copy()
theta = gradient_descent(X_with_bias, Y_s, theta, learning_rate, 2)
print("Updated theta after two iterations of gradient descent:")
print(theta)

plot_model(X_matrix, theta, title="After 2 Iterations")

#---------------------------------------------------------------------------------------
# 7. Linear regression + gradient descent - n iterations
theta = starting_theta.copy()
iterations = 100
theta, costs = gradient_descent(X_with_bias, Y_s, theta, learning_rate, iterations, save_costs=True)

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
plot_model(X_matrix, theta, title=f"After {iterations} Iterations")


# 8. Calculating MSE cost of least squared closed form solution
final_cost_ls = mse_cost_function(X_with_bias, Y_s, ls_theta)
print(f"Final cost of least squares closed form solution: {final_cost_ls}")

#---------------------------------------------------------------------------------------
# A.II.
#---------------------------------------------------------------------------------------

# polynomial regression

def quartic_cost(X, Y, theta):
    n = len(Y)
    pred_X = X @ theta
    cost = (1/(4*n)) * np.sum(np.power(Y - pred_X,4))
    return cost

def quartic_gradient(X, Y, theta):
    n = len(Y)
    pred_X = X @ theta
    gradient = - (1/n) * (X.T @ np.power((Y - pred_X),3))
    return gradient

rng = np.random.default_rng(SEED)
theta = rng.standard_normal(3)
print(f"Starting theta for polynomial regression: {theta}")
learning_rate = 0.01
iterations = 5000

X = np.column_stack([np.ones(n), X_s, X_s**2])
theta, costs = gradient_descent(X, Y_s, theta, learning_rate, iterations, cost_function=quartic_cost, cost_gradient=quartic_gradient, save_costs=True)

# plot the costs over iterations to visualize convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# normal Plot
ax1.plot(costs)
ax1.set_title(f"Costs over {iterations} iterations")
ax1.set_xlabel("Iteration number")
ax1.set_ylabel("Cost value")

# logarithmic Plot
ax2.plot(costs)
ax2.set_yscale('log')
ax2.set_title(f"Costs over {iterations} iterations (Log Scale)")
ax2.set_xlabel("Iteration number")
ax2.set_ylabel("Cost value (Log)")

plt.tight_layout()
plt.show()

print(f"Final cost: {costs[-1]}")

# plot the final model after n iterations
print(f"Updated theta after {iterations} iterations of gradient descent:")
print(theta)

X_matrix = np.c_[np.ones(len(x_sorted)), x_sorted, x_sorted**2]
plot_model(X_matrix, theta)