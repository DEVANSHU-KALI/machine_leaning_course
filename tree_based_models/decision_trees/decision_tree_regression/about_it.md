# Decision trees for regression
- When the target variable is continuous (e.g., predicting house prices or temperatures), decision trees are used for regression.
    - The standard algorithm here is CART (Regression mode).
    - Instead of entropy or Gini, CART uses variance reduction (or standard deviation reduction) to decide splits.
    - At each step, the algorithm chooses the feature that most reduces the variability of the target values in the child nodes compared to the parent node. This ensures that predictions at the leaves are as accurate as possible by minimizing prediction error.
- as you can see that the target attribute in the classification example (play tennis) has the only two values, yes or no, which is discrete. But here we are going to have real numbers 
## The whole worflow using a example 
- 