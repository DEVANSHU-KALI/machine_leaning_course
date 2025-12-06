import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = load_diabetes()
X = data.data   # features (442 samples, 10 features)
y = data.target # target (continuous)

m, n = X.shape
X = np.c_[np.ones(m), X]   # add intercept column
n = X.shape[1]             # parameters (intercept + features)

# Initialize parameters
theta = np.zeros(n)
alpha = 0.01
iterations = 1000
losses = []

# Batch Gradient Descent starts here---
for _ in range(iterations):
    y_pred = X.dot(theta)              # predictions
    error = y_pred - y                 # residuals
    gradients = (1/m) * X.T.dot(error) # average gradient
    theta -= alpha * gradients         # update step
    loss = (1/m) * np.sum(error**2)    # MSE
    losses.append(loss)
#--------------ends here--------------

# Final predictions (compute using the final theta)
y_pred = X.dot(theta)

# Regression metrics (use correct function signatures)
print(f'mse:{mean_squared_error(y, y_pred):.4f}')
print(f'mae:{mean_absolute_error(y, y_pred):.4f}')
print(f'r_squared:{r2_score(y, y_pred):.4f}')
print("Final coefficients:", theta)

# Plotting 
plt.plot(losses)
plt.grid(True)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Convergence of Batch Gradient Descent (Diabetes dataset)")
plt.show()
