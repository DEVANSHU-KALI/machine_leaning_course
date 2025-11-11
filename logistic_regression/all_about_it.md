## Core of the logistic regression model
This model is built on the linear regression, means both are similar. Instead of just predicting continues values, we apply the sigmoid function to the linear output and get probabilities between 0 and 1. The result is a smooth sigmoid curve which models the likelihood of class membership in the binary classification. 

The line **"models the likelihood of the class membership"** means:

1) models the likelihood...: the model doesn’t say this specific output *belongs* to a class; it gives the *likelihood* (there is a specific chance that this output belongs to a particular class).  
   For example: if the output is 0.95, we say there is a 95% chance that it belongs to class 1.

2) class membership: as this is a classification model, there will be different classes, so an output belongs to some class — either 1 or 0, as an example.

## how does the model work
1) the whole process is same as the linear regression,  