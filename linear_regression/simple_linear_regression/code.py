import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])       # Example: Feature (e.g., study hours)
Y = np.array([3, 4, 2, 5, 6])       # Example: Target (e.g., test score)

#parameters
learning_rate = 0.01     
iterations = 1000         
m, c = 0, 0       

n = float(len(X))

for i in range(iterations):
# Gradient Descent Algorithm working starts here ----------------
    Y_pred = m * X + c

    # Calculate gradients
    m_gradient = (-2/n) * np.sum(X * (Y - Y_pred))
    c_gradient = (-2/n) * np.sum(Y - Y_pred)

    #updating the gradients
    m = m - learning_rate * m_gradient
    c = c - learning_rate * c_gradient
#------------ends here-------------------------------------------

# Final parameters
print(f"\nFinal parameters: m = {m:.4f}, c = {c:.4f}")

# Cost Function (Mean Squared Error)
def cost_function(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Compute and display final cost
final_loss = cost_function(Y, m * X + c)
print(f"Final Loss: {final_loss:.4f}")

#plotting the graph with the regession line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Fitted Line')
plt.show()