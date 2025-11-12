## Core of the logistic regression model
This model is built on the linear regression, means both are similar. Instead of just predicting continues values, we apply the sigmoid function to the linear output and get probabilities between 0 and 1. The result is a smooth sigmoid curve which models the likelihood of class membership in the binary classification. 

The line **"models the likelihood of the class membership"** means:

1) models the likelihood...: the model doesn’t say this specific output *belongs* to a class; it gives the *likelihood* (there is a specific chance that this output belongs to a particular class).  
   For example: if the output is 0.95, we say there is a 95% chance that it belongs to class 1.

2) class membership: as this is a classification model, there will be different classes, so an output belongs to some class — either 1 or 0, as an example.

## how does the model work
1) we take the data, as the first thing of the process.
- first difference is the formula of the model which is: y_pred=𝜎(z)=1/1+e^-z where z=m*x+c
   - y_perd is the predicted value and the z is sigmoid function 𝜎(𝑧) which squashes the output between 0 and 1.
2) we calculate the loss using the loss fuction, which is a common step done. from many of the loss functions here we use the 
binary cross entropy (log loss). 
   - the loss function: loss=-[y.log(y_pred)+(1-y).log(1-y_pred)], **where y is the actual value and y_pred is the predicted value.** 
   - reason behind using this specific loss function as default:
      - as we are predicting the probabilities, log loss measure how well those probabilities match those true labels.
      - log loss is defined from the bernoulli distributed outcomes, so minimizing the loss means mazimizing the likelihood.
      - log loss is convex,so that the gradients can easily find the global minimum.
      - mse treate the errors linerly, which dosent make sense in the classification process.
      - log loss focuses on how condifent the model is, and that the crusial thing here.
           
3) now comes the optimization step.
- this step is common and same as it is done in the linear regression. 

4) example image of the logistic regression
<!-- fixed path: use the actual filename in ../images/ -->
![photo](../images/example_image_logisitic_reg.png)