# What is it?
- polynomial regression is a type of regression where the input features are the powers of input feature itself.
- lets say you have some features like x1, x2, x3, now the features would be like x1, x1^2, x1^3, x2, x3. the reason behind this lies the main thing why this model is used. 
- polynomial regression is not actually called polynomila linear regression sometimes, because we dont always get a stright line like other liner models in this model, we can also get a curve or the line bending to other side, even trying to adapt to not linear data and get good predictions, although it comes under linear models because the equation of this model is linear in parameters. 
- this model is first model you are going to learn in the intial stages of machine learning where you see the line bending according the data points.
## Why do we even take the powers of the features
- to capture the non linear pattern in the data. This make the line to bend and go through the points. 
## Why only training the powers of x1, why not the other features 
- it depends upon which features is showing no linear patterns in the data, if its x1 showing the non linear patterns in the data, we take the powers of it. 
## Disadvantages of this model
## Possible questions about this model and answers
- **How to know that, the feature is showing non linearity in the data?**
    - the easiest way to know that is to, __plot that feature with the target term.__ 
    - the other method is called __persons corellations coefficitent(r)__: you calculate this between the feature and the target value
        - if the value close to 1 or -1 indicates a strong linear relationship and a value close to 0 tell the feature is showing non linear pattern.
- **What if we have some 100s of features, what is the optimal way to know which feature should we use for powering?**
    - one of the most optimal way to know is using the k-folds cross validation, although its a advanced topic.
    - but in general this model is better for konnwoing the existence of the model, becuase in paractical we just dont use this model often wehn we have some 100 features, due to disadvantages of this model.
- **What about the other features, are all the features needed even if we are adding some new feature columns?**
    - you take the powers to the features which are showing non linear relation pattern, and if you feel like other features are also important, you can keep them for training the model, or remove them if you dont want.
- **Is there any limit to the degree we take?**
    - yes there is a limit like, if you have n features, you can take the degree up to n-1, but its not recomeneded in machine learning practice, due to the problems it may create.
    - lets say you took the degree till n-1, what happens if you plot it is, you end up with a line which perfectly fits your data, it goes through every point which bring the problems like **over fitting**, **extrapolation problem** like going to higher degree makes the model unpredictable outside the training data.
- **How is the equation gonna look if we want more than one feature's powers?**
- **What happens if we take more degree?**
    - it may increse **over fitting**,  **extrapolation problem** (explained above), the model also captures the noise in the data (if you take higher degree the line gets more flexible like a band and tries to go through every data point and while in that processs, it also goes through the nose and captures that too,,, which is not good for a model.) 
- **You may also here that, you only take the feature and its powers in the model and do the predictions, but what about the other features?**
    - there is no restriction like you should only take powers of one fetaure, its up to you, if you feel like 2 features are showing non linear relation, you can take the powers of that feature too. all that matters is, till how much degree you going to take it to.
- **Why is this model not used wide and only heard just in the theory?**