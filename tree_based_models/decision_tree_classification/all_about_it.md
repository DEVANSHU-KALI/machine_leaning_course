Now as we discussed, tree based models are used for both classification and regression.
- here's about the **classification**
# What is it?
Decision trees for classificaionw work by **recursively splitting the data into subsets** using the feature-based rules which minimize class purity. each split is choosen using metrics like **Entropy** or **Gini index**.

## How it actuall works
- Root node: entire dataset
- Decision nodes: feature_based questions (e.g., age>30)
- Branches: outcomes of the questions
- Leaf node: final class label (e.g., yes=buyer, no=not a buyer)

