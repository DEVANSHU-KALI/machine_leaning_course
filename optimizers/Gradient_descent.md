# What is Gradient Descent

- *It's an iterative optimization algorithm which is used to train the machine learning models to adjust the parameters and minimize the loss function.*
- You can also say it like, its a **first order** optimization algorithm to adjust the parameters to get the minimum loss.
  - You'll later know why I used the term first order there.

# How it optimizes the loss !!

- Mainly there are three types of this, and three of them work differently.
- It initially computes the gradients and later updates them eventually. It stop until it reaches a stopping condition which are maximum iteration, convergence or lack of improvements etc.

## Types of gradient descent:

1) Stochastic: Takes only single sample at a time to calculate the gradients.
2) Batch: Uses the whole dataset to calculate the gradients.
3) Mini batch: You create batches and and for every set of batch you calculate the gradients and update them.
- You can see, how the three work differently below.

## Process to see how they work in the code.
