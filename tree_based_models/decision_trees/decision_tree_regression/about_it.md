# Decision trees for regression
- When the target variable is continuous (e.g., predicting house prices or temperatures), decision trees are used for regression.
    - The standard algorithm here is CART (Regression mode).
    - Instead of entropy or Gini, CART uses variance reduction (or standard deviation reduction) to decide splits.
    - At each step, the algorithm chooses the feature that most reduces the variability of the target values in the child nodes compared to the parent node. This ensures that predictions at the leaves are as accurate as possible by minimizing prediction error.