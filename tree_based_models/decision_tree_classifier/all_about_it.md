Now as we discussed, tree based models are used for both classification and regression.
- here's about the **classification**
# What is it?
Decision trees for classificaionw work by **recursively splitting the data into subsets** using the feature-based rules which minimize class purity. each split is choosen using metrics like **Entropy** or **Gini index**.

## How it actuall works
- Root node: entire dataset
- Decision nodes: feature_based questions (e.g., age>30)
- Branches: outcomes of the questions
- Leaf node: final class label (e.g., yes=buyer, no=not a buyer)

## Mathematical formulas
- entropy
- information gain
- gini impurity (alternative metric) 
- note: as we cant display the formal here, find it online 

## Advantages and disadvantages
- Easy to interpret	                       | Can overfit if too deep
- Handles categorical & numerical data     | Sensitive to small data changes
- No assumptions about distribution	Biased | toward features with many levels

## more about the code
- from here understandig new concepts help a lot in the coding like, using datsets, data transformation, data visualization, evaluation metric, splitting data 

## Example 
- lets take the data set like this 
id	Age	Student	Buys
1	22	Yes 	Yes
2	25	Yes 	Yes
3	47	No	    No
4	52	No  	No
5	46	Yes 	Yes
6	56	No  	No
