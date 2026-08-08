# What is it?

- Polynomial regression is a type of regression where the input features include powers of the original feature(s).
- Let's say you have features like `x1`, `x2`, `x3`. The transformed features might be `x1`, `x1^2`, `x1^3`, `x2`, `x3`. This lets the model capture nonlinear relationships in the data.
- Polynomial regression is sometimes called polynomial linear regression. We don't always get a straight line like other linear models; we can get a curve (the line bends) to adapt to nonlinear data while still being a linear model because the equation is linear in the parameters.
- This model is one of the first you learn in the initial stages of machine learning, where you can see the line bending according to the data points.

## Model Equation

- `𝑌 = 𝑚1⋅𝑋 + 𝑚2⋅𝑋^2 + 𝑚3⋅𝑋^3 + ⋯ + 𝑚𝑑⋅𝑋^𝑑 + 𝑐`

## Why Take Powers of the Features?

- To capture nonlinear patterns in the data. This makes the model more flexible and better able to fit the points.

## Why Train Powers of Only `x1` and Not Other Features?

- It depends on which features show nonlinear patterns. If `x1` shows nonlinearity, we take powers of it.

## Disadvantages of This Model

- The main disadvantage is known as **overfitting**.
- **Multicolinearity**: Powers of the same feature are highly correlated, which makes coefficient estimates unstable.
- **Scalability issues**: Works well with small datasets and low degrees; not suitable for large datasets.
- **Limited flexibility compared to modern models**: Tree-based, ensemble, and deep learning models often perform better.
- **Rarely used in practice**: Many real-world problems prefer gradient boosting, random forests, neural networks, etc.
