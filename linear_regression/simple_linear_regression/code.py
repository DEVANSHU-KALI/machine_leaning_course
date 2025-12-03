#Note: this code uses batch gradient descent.
import numpy as np
import matplotlib.pyplot as plt
# for evaluations of the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X = np.array([1.0, 1.5, 2.0, 2.5, 3.0,
              3.5, 4.0, 4.5, 5.0, 5.5,
              6.0, 6.5, 7.0, 7.5, 8.0,
              8.5, 9.0, 9.5, 10.0, 10.5])          # Feature (strudied hours )

Y = np.array([2.3, 3.1, 4.2, 4.8, 6.1,
              7.0, 8.2, 9.0, 10.1, 11.2,
              12.0, 13.5, 14.1, 15.0, 16.3,
              17.2, 18.1, 19.0, 20.5, 21.2])       # Target (e.g., test score)

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

# metrices
Y_pred=m * X + c
print(f'MSE:{mean_squared_error(Y,Y_pred):.4f}')
print(f'MAE:{mean_absolute_error(Y,Y_pred):.4f}')
print(f'R_squared:{r2_score(Y,Y_pred):.4f}')

#plotting the graph with the regession line
plt.grid(True)
plt.title('Linear regression using batch gadient descent')
plt.xlabel('studied hours (X)')
plt.ylabel('test score (Y)')
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, m * X + c, color='red', label='Fitted Line')
plt.legend()
plt.show()