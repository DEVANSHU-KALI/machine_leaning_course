# Decision trees as classification
- When the target variable is categorical (e.g., “spam” vs “not spam”), decision trees are used for classification.
    - ID3 algorithm builds trees using Information Gain based on entropy.
    - C4.5 improves on ID3 by handling continuous attributes, missing values, and pruning, and it uses Gain Ratio as its splitting metric.
    - CART (Classification mode) is another algorithm that uses the Gini Index to measure impurity. In all cases, the goal is to choose splits that maximize class purity in the resulting subsets.