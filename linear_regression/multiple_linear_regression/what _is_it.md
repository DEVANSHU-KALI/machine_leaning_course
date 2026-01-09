# What is this
**Multiple linear regression models the relationship between one dependent variable and two or more independent variables.** It allows using multiple features to predict the output.

- Instead of taking only one feature for X, we take more than one here; that's what makes this model different from simple linear regression.
- **Model equation**: y_pred = m1·X1 + m2·X2 + m3·X3 + ... + c
- Coefficients are usually denoted by β (beta), not m.
- Here m1 is the coefficient of X1 (feature 1), m2 is the coefficient of X2 (feature 2), and so on.
- The number of coefficients equals the number of features (plus the intercept).
- We can fit this model using the closed-form normal equation or iterative optimizers (e.g., gradient descent). The closed-form solution doesn't require iterative optimization; it directly computes the parameter values, similar to simple linear regression.

## Points to know
1) You can't easily visualize the regression hyperplane in 2D when there are more than one feature; for two features you can use a 3D plot. In code examples we often plot the training loss to observe learning progress.

### Parts of the code explained
1) Dataset:
    - We use the diabetes dataset from sklearn. It contains 442 samples and 10 features, which is suitable for multiple linear regression examples.
2) X = np.c_[np.ones(m), X] (why this line)
    - This line adds a column of ones to X so the model can learn an intercept term instead of being forced through the origin.
    - n = X.shape[1] gives the number of parameters (including the intercept).
3) theta = np.zeros(n) initializes all parameters to zero for gradient descent.
4) alpha is the learning rate.
5) iterations is the number of gradient descent steps.
6) losses is a list to store the loss value at each iteration.

