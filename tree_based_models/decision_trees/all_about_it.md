# What is it?
- Decision trees are a machine learning concept used for predictive modeling. They represent decisions in a tree-like structure, where each internal node corresponds to a feature-based test, each branch represents the outcome of that test, and each leaf node gives a final prediction. The idea is to recursively split the dataset into subsets until the data in each subset is as “pure” or homogeneous as possible with respect to the target variable.
- We can use this concept for classification and regression.

## How It Actually Works
- Root node: Entire dataset
- Decision nodes: Feature_based questions (e.g., `age > 30`)
- Branches: Outcomes of the questions
- Leaf node: Final class label (e.g., `yes = buyer`, `no = not a buyer`)

## Mathematical Formulas
- Entropy
- Information Gain
- Gini Impurity (alternative metric) 
- Note: As we cant display the formula here, find it online 
