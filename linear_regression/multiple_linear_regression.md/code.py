import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Generate synthetic dataset
# -----------------------------
np.random.seed(42)
m = 100  # number of samples
x1 = np.random.uniform(0, 10, m)   # feature 1
x2 = np.random.uniform(0, 5, m)    # feature 2

# True relationship: y = 2 + 1.5*x1 + 0.7*x2 + noise
y = 2 + 1.5*x1 + 0.7*x2 + np.random.normal(0, 1, m)

# -----------------------------
# 2) Prepare design matrix
# -----------------------------
X = np.column_stack((x1, x2))
X = np.c_[np.ones(m), X]   # add intercept column
n = X.shape[1]             # number of parameters (intercept + features)

# -----------------------------
# 3) Initialize parameters
# -----------------------------
theta = np.zeros(n)
alpha = 0.01
iterations = 1000
losses = []

# -----------------------------
# 4) Batch Gradient Descent
# -----------------------------
for _ in range(iterations):
    y_pred = X.dot(theta)              # predictions
    error = y_pred - y                 # residuals
    gradients = (1/m) * X.T.dot(error) # average gradient
    theta -= alpha * gradients         # update step
    loss = (1/m) * np.sum(error**2)    # MSE
    losses.append(loss)

# -----------------------------
# 5) Results
# -----------------------------
print("Final coefficients:", theta)
print("Final loss: {:.4f}".format(losses[-1]))

# -----------------------------
# Regression metrics
# -----------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot

def adj_r2(y_true, y_pred, p, m):
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (m - 1) / (m - p - 1)

# After training (when you have y_pred and theta):
print("MSE: {:.4f}".format(mse(y, y_pred)))
print("R²: {:.4f}".format(r2_score(y, y_pred)))
print("Adjusted R²: {:.4f}".format(adj_r2(y, y_pred, p=X.shape[1]-1, m=len(y))))

# ----------------------------
# 6) Plot convergence
# -----------------------------
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Convergence of Batch Gradient Descent")
plt.show()
