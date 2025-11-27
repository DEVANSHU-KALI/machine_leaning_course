**linear regression** is a supervised machine learning model which models the relationship between 1 dependent variable(also called as target variable) and 1 or more independent variables(features).

- There are 3 types of linear regression
    - simple linear regression
    - multiple linear regression
    - polynomial linear regression
- These three work on the same concept but work differently.

## How this linear regression algorithm works
1) This algorithm fits a straight line(regression line) through the data points, and later you can make the predictions.
    - the basic formula for this linear regression looks like y_pred = m*x + c
        - m=slope (coefficient): determining how much the y changes per unit change in x
        - c=intercept (bias): value of y when x=0 
            - It shifts the line up or down on the graph.
            - If c=5, then even the x=0, the predicted value of y will be 5.
2) We calculate the loss using loss functions, where loss stands for the difference between the actual and predicted value, which should be as small as possible to make a good  prediction. There are multiple loss functions, but here we are going to use the 
       MSE (mean squared error) as an example.
3) Now comes the optimization step.
    - There are some types of optimizers, you can also choose them, but what it does is reduce the loss by adjusting the  parameters step-by-step.
    - at each step, the loss is reduced.
    - this process continues until the loss is as low as possible or meets a stopping condition.
    - You calculate the gradients for both m and c, and then update the gradient, which I'll show in the code part how it's done.

## applications
    - house price prediction

### sturcture to start
- I'm metioning this because, you should start with the simple linear regression first and the structure of git showing the file is different if you ever noticed that, it shows which is created recently and as you can see you would see other types prioritized before the simple linear regression. so let's start
- Simple linear regression first
- Mulitple LR
- Polynomial LR
- Ridge LR
- Lasso LR
- Elastic net LR 
    

    
