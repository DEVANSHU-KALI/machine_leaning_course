Note: This page assumes you have read the linear regression overview.

# What does it mean?
- Simple linear regression models the relationship between one dependent variable and one independent variable.
- Formula: y_pred = m * x + c
- See "All_about_linear_regression.md" for definitions of m and c.
- As explained on that page, the next step is optimization, which you can observe in the code.
- For more details about optimization, see the gradient descent page in the optimizer folder. The same optimization process applies to this model; it is used there as an example to explain the optimizer.
- You can run the code to see how the algorithm converges and view the resulting plot.

## How to execute the code
- Install required packages:
  - python -m pip install numpy matplotlib
- Copy the code into your IDE or run the script in a Python interpreter. The script prints the final parameters and displays a graph.
- The provided implementation uses batch gradient descent, an iterative approach to find the best-fit line.
- You can also use the closed-form Normal Equation to compute parameters without iteration (efficient for small datasets). Alternatively, use sklearn's implementation:
  - from sklearn.linear_model import LinearRegression
  - This typically results in much shorter code.