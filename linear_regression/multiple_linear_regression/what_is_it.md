# What is this

**Multiple linear regression models the relationship between one dependent variable and two or more independent variables.** It allows using multiple features to predict the output.

- Instead of taking only one feature for `X`, we take more than one here; that's what makes this model different from simple linear regression.
- **Model equation**: `y_pred = m1·X1 + m2·X2 + m3·X3 + ... + c`
- Coefficients are usually denoted by β (beta), not `m`.
- Here `m1` is the coefficient of `X1` (feature 1), `m2` is the coefficient of `X2` (feature 2), and so on.
- The number of coefficients equals the number of features (plus the intercept).
- We can fit this model using the closed-form normal equation or iterative optimizers (e.g., gradient descent). The closed-form solution doesn't require iterative optimization; it directly computes the parameter values, similar to simple linear regression.

## Points to know

1) You can't easily visualize the regression hyperplane in 2D when there are more than one feature; for two features you can use a 3D plot. In code examples we often plot the training loss to observe learning progress.


