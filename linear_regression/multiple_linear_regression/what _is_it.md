# what is this
**Mutltiple linear regression models relationship between one dependent variable and two or more independent variables**. so that we can use multiple features to predict the output.
- instead of taking only 1 feature for the X variables, we take more than 1 here, that's what make this model different from others.
- **Formula**: y_perd=m1X1+m2X2+m3X3...+c
- the coeffecients are actually represented as symbol beta not m, 
- where m1 is the coefficient of X1 (feature 1), m2 is the coeffecient of X2 (feature 2), so on.
- it all depends on how many features you have, number of features, same number of coeffecients. 
- we can execute this model using the closed form solution and also using the optimizers. closed form solution dosent require any optimization, it directly computes the values and predition is done. same as we talked in the simple linea regression.

## points to know 
1) you can't actually plot the regression line on the 2d plane, you need 3d plotting to see the regression line. But here in the code we are just plotting the loss to see how the model is learning. 
2) 

### parts of the code explained
1) dataset:
    - here we are taking a diabetes dataset, which is available in the sklearn library. which has 422 samples with 10 features because we are working with mulitlinear regression.
2) X = np.c_[np.ones(m), X] (why this line)
    - this line adds a column of onces to the x to not force the model to start thorugh the origin. 