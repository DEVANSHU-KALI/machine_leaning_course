# What is it
- A **Random Forest** is an ensemble of multiple decision trees. For classification, each tree votes for a class and the forest's prediction is the majority vote; for regression, the forest typically averages the outputs. Think of it like electing a chairperson by majority vote—each voter (tree) casts a vote and the most common choice wins.

## What makes this model better than a single decision tree
- Single decision trees are sensitive to the training data: small changes can lead to very different trees (high variance). Random forests reduce variance by averaging many de-correlated trees (built using bootstrapped samples and random feature subsets), which usually results in more robust and generalizable predictions.
- 