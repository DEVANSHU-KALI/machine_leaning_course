# What is it?
- Polynomial regression is a type of regression where the input features include powers of the original feature(s).
- Let's say you have features like x1, x2, x3. The transformed features might be x1, x1^2, x1^3, x2, x3. This lets the model capture nonlinear relationships in the data.
- Polynomial regression is sometimes called polynomial linear regression. We don't always get a straight line like other linear models; we can get a curve (the line bends) to adapt to nonlinear data while still being a linear model because the equation is linear in the parameters.
- This model is one of the first you learn in the initial stages of machine learning, where you can see the line bending according to the data points.

## Model equation
- 𝑌 = 𝑚1⋅𝑋 + 𝑚2⋅𝑋^2 + 𝑚3⋅𝑋^3 + ⋯ + 𝑚𝑑⋅𝑋^𝑑 + 𝑐

## Why take powers of the features?
- To capture nonlinear patterns in the data. This makes the model more flexible and better able to fit the points.

## Why train powers of only x1 and not other features?
- It depends on which features show nonlinear patterns. If x1 shows nonlinearity, we take powers of it.

## Disadvantages of this model
- The main disadvantage is known as **overfitting**.
- **Multicollinearity**: powers of the same feature are highly correlated, which makes coefficient estimates unstable.
- **Scalability issues**: works well with small datasets and low degrees; not suitable for large datasets.
- **Limited flexibility compared to modern models**: tree-based, ensemble, and deep learning models often perform better.
- **Rarely used in practice**: many real-world problems prefer gradient boosting, random forests, neural networks, etc.

## Possible questions and answers

- **How can you tell a feature is showing nonlinearity?**
    - The easiest way is to plot the feature against the target.
    - Another method is Pearson's correlation coefficient (r): values close to ±1 indicate a strong linear relationship; values near 0 suggest the relationship may be nonlinear.

- **If we have hundreds of features, how do we know which to raise to powers?**
    - A practical approach is to use k-fold cross-validation (an advanced topic).
    - In practice, polynomial regression is mainly useful for small or simple problems rather than large feature sets.

- **Are all features needed if we add polynomial terms?**
    - You typically add polynomial terms for features that show nonlinear relationships. Keep features that are important; remove uninformative ones.

- **Is there a limit to the degree we should take?**
    - Very high degrees lead to overfitting and extrapolation problems. Increasing degree may make the model fit training data perfectly but perform poorly on new data.

- **How does the equation look if we use powers of more than one feature?**
    - For features x1, x2, x3, if you take powers of x1 and x3, the feature set might be: x1, x1^2, x1^3, x2, x3, x3^2.

- **What happens if we take a higher degree?**
    - It may increase overfitting and extrapolation issues. The model can also capture noise in the data.

- **Do we have to take powers of only one feature?**
    - No. You can take powers of any features that show nonlinear relationships. The important question is how high a degree you choose.

- **Why isn't this model widely used in practice?**
    - Mainly because of its disadvantages (overfitting, multicollinearity, scalability). It's still useful to learn, as it illustrates how models can bend to fit data.