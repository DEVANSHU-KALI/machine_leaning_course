# Polynomial regression (degree 3) using batch gradient descent on a built-in sklearn dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- load a built-in regression dataset (diabetes) and pick one feature for easy plotting ---
data = load_diabetes()
X_raw = data.data[:, 2]          # pick BMI-related feature (1D) for clarity
y = data.target

# reshape and create polynomial features (degree = 3): [x, x^2, x^3]
X = X_raw.reshape(-1, 1)
degree = 3
X_poly = np.hstack([X**i for i in range(1, degree + 1)])  # shape (n_samples, degree)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# feature scaling (mean 0, std 1) done inline
mean_X = X_train.mean(axis=0)
std_X = X_train.std(axis=0, ddof=0)
std_X[std_X == 0] = 1.0
X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X

# parameters for batch gradient descent
lr = 0.01
iterations = 5000
n_train = X_train_scaled.shape[0]

# initialize weights and bias
w = np.zeros(X_train_scaled.shape[1])   # one weight per polynomial term
b = 0.0

# batch gradient descent loop
for it in range(iterations):
    y_pred = X_train_scaled.dot(w) + b
    error = y_train - y_pred

    # gradients
    w_grad = (-2.0 / n_train) * (X_train_scaled.T.dot(error))
    b_grad = (-2.0 / n_train) * np.sum(error)

    # update
    w = w - lr * w_grad
    b = b - lr * b_grad

# final parameters
print(f"Final weights: {w}")
print(f"Final bias: {b:.6f}")

# predictions
y_train_pred = X_train_scaled.dot(w) + b
y_test_pred = X_test_scaled.dot(w) + b

# print metrics (using sklearn functions directly)
print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")

print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"Test R2: {r2_score(y_test, y_test_pred):.4f}")

# plotting (original feature vs target and fitted polynomial curve)
# prepare a smooth curve for plotting the polynomial fit
x_plot = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)
x_plot_poly = np.hstack([x_plot**i for i in range(1, degree + 1)])
x_plot_scaled = (x_plot_poly - mean_X) / std_X
y_plot = x_plot_scaled.dot(w) + b

plt.figure(figsize=(8, 5))
plt.scatter(X_raw, y, color='lightgray', label='All data')
plt.scatter(X_train[:, 0], y_train, color='blue', label='Train')
plt.scatter(X_test[:, 0], y_test, color='green', label='Test')
plt.plot(x_plot, y_plot, color='red', linewidth=2, label=f'Polynomial degree {degree} fit')
plt.xlabel('Selected feature (raw)')
plt.ylabel('Target')
plt.title('Polynomial Regression (batch gradient descent) on sklearn diabetes dataset')
plt.legend()
plt.grid(True)
plt.show()
