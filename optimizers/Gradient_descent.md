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

- I'll mention how all these work in the code separately, but before knowing how they work, you need to know how the process starts and the key points and variables needed for that. If you see the code you'll get some idea about the structure. Now let's start
- The variables common for three of these are as follow:
  - Learning rate: You give a small value for this and in the codes we give it a 0.1.
  - `m` and `c`: We initially give them both zeros.
  - `n`: Length of the `x`.
  - Iterations: We gave 1000.
- I want you to see the code of simple linear regression once to see from where the optimizer part starts. And I've also mentioned that part clearly.
- Now let's see how those work.
- **Note**: To direct implement these into the code, remove the lines explaining the process and directly replace the gradient calculation part with these, the other part is same for all.

### 1) How **Batch gradient descent** works in the code.

- The above mentioned variables remain same.
- You run a loop for iterations:
  ```python
  Y_pred = m * X + c
  m_gradient = (-2/n) * np.sum(X * (Y - Y_pred))
  c_gradient = (-2/n) * np.sum(Y - Y_pred)
  ```
  - Note: The `np.sum` indicates that we are taking the whole dataset for each iteration to calculate the gradients and update them.
- Now the updating part:
  ```python
  m = m - learning_rate * m_gradient
  c = c - learning_rate * c_gradient
  ```
