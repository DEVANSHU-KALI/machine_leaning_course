# What are optimizers

- You can say them like algorithms which **reduce the loss by adjusting the model parameters through multiple iterations**.
- There many types of these which are used according to the situation.

# Why are these actually used ?

- These are used in the process to find the optimal parameters for better predictions.

# Can't we run the model without these?

- Yes, but only in some special cases.
  - For some models like linear regression, there's closed form solution using the matrix algebra, in such cases you dont need any iterative approach, the best parameters are computed automatically.
- For most ml models (dl and large datasets), closed form doesn't exist. So optimizer come into play.
- These optimizers iteratively adjust the parameters to reduce the loss as in the first point.
