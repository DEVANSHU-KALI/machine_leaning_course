# what is it:
- polynomial regression is a type of regression where the input features are the powers of input feature itself.
- lets say you have some features like x1, x2, x3, now the features would be like x1, x1^2, x1^3, x2, x3. the reason behind this lies the main thing why this model is used. 
- polynomial regression is not actually called polynomila linear regression sometimes, because we dont always get a stright line like other liner models in this model, we can also get a curve or the line bending to other side, even trying to adapt to not linear data and get good predictions, although it comes under linear models because the equation of this model is linear in parameters. 
- this model is first model you are going to learn in the intial stages of machine learning where you see the line bending according the data points.
## why do we even take the powers of the features
- to capture the non linear pattern in the data. This make the line to bend and go through the points. 
## why only taing the powers of x1, why not the other features 
- it depends upon which features is showing no linear patterns in the data, if its x1 showing the non linear patterns in the data, we take the powers of it. 
## possible questions about this model
- how to know that, the feature is showing non linearity in the data?
- what if we have some 100s of features, what is the optimal way to know which feature should we use for powering?
- what about the other features, are all the features needed even if we are adding some new feature columns?
- is there any limit to the degree we take?
- how is the equation gonna look if we want more than one feature's powers?
- what happens if we take more degree?
- what are the disadvantages of this model?
