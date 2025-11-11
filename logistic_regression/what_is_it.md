## Core of the logistic regression model
This model is built on the linear regression, means both are similar. Instead of just predicting continues values, we apply the sigmoid function to the linear output and get probabilities between 0 and 1. The result is a smooth sigmoid curve which models the likelihood of class membership in the binary classification. 

the line 'models the likelihood of the class memebership' means:
    1. models the likelihood.... :the model dosent say this specific output belongs to this class, it gives the likehood (there  
       is a specific chance that this output belongs to specific class). for example: if the output is 0.95, we say there is a chance that this belongs to class 1. 
    2. class memebership: as this is a classification model, there will be different classes, so a output belongs to some class, 
       either 1 or 0 as an example.

