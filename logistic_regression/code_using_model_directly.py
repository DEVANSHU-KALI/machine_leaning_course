import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simple dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # hours studied
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 0 = fail, 1 = pass

# Train logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
y_prob = model.predict_proba(X)[:, 1]  # probability of class 1

plt.scatter(X, y, color='red', label='Actual')
plt.plot(X, y_prob, color='blue', label='Predicted Probability')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression Sigmoid Curve')
plt.legend()
plt.grid(True)
plt.show()

#predicting the class of new value as integer
print(model.predict([[11]])[0])

#predicting the probability of new value as float
print(model.predict_proba([[11]])[0][1])