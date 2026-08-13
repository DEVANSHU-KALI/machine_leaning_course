# Decision Trees for Regression
- When the target variable is continuous (e.g., predicting house prices or temperatures), decision trees are used for regression.
    - The standard algorithm here is CART (Regression mode).
    - Instead of entropy or GINI, CART uses variance reduction (or standard deviation reduction) to decide splits.
    - At each step, the algorithm chooses the feature that most reduces the variability of the target values in the child nodes compared to the parent node. This ensures that predictions at the leaves are as accurate as possible by minimizing prediction error.
- As you can see that the target attribute in the classification example (play tennis) has the only two values, yes or no, which is discrete. But here we are going to have real numbers.

### NOTE: All the images regarding this explanation will be inside the `DTR_images` (Decision Tree Regression images) folder of `images` folder. So that I need not to mention.

## The Whole Workflow Using a Example 
- Lets take a dataset which has 5 attributes (columns) `'outlook'`, `'temp'`, `'humidity'`, `'wind'` and `'golf players'` (target attribute).
- In the classification we took the information gain to make splits, but here we are taking **standard deviation reduction to make splits**. So the workflow goes as follow.
- First we need to calculate the standard deviation (SD) for the whole dataset (i.e. SD of target attribute), next the reduction in SD for each of the other individual attributes, based on that reduction we take the specific attribute and start building the tree. So, SD of the target attribute = 9.32 if you calculate.
