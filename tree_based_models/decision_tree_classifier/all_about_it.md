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
- | ID | Age | Student | Buys |
- |----|-----|---------|------|
- | 1  | 22  | Yes     | Yes  |
- | 2  | 25  | Yes     | Yes  |
- | 3  | 47  | No      | No   |
- | 4  | 52  | No      | No   |
- | 5  | 46  | Yes     | Yes  |
- | 6  | 56  | No      | No   |
- now as there are three no's and three yes's the entropy is maximal
- entropy and information gain (formulas).
    - entropy of a set S with class proportions pi is 
    - 𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝑆)=−∑ with lower limit(𝑖) 𝑝𝑖log2𝑝𝑖
- information gain on splitting an attribut A is
    - 𝐼𝐺(𝑆,𝐴)=𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝑆)−∑ with lower limit (𝑣∈𝑉𝑎𝑙𝑢𝑒𝑠(𝐴)) (∣𝑆𝑣∣/∣𝑆∣)𝐸𝑛𝑡𝑟𝑜𝑝𝑦(𝑆𝑣)
- these are standard definations for choosing the splits.
- Root entropy.
    - two classes with equal proportions Pyes = Pno = 0.5
- candidate split 1 - strudent(Yes / No)
    - student = yes:ids 1,2,5 -> all yes -> entropy = 0
    - student = no:ids 3,4,6 -> all no -> entropy = 0 
    - weighted entropy:
        - (3/6) 0 + (3/6) 0 = 0......, thats because, we have 3 out 6 as yes and 3 out of 6 as no, so we got 3/6 two times and mulitiplied with zero is their entropies
        - so IG(S,Student)= 1.0 - 0 = 1.0 
        - this is perfect split which yeilds the pure leaves.

## points to note:
- from this stage learning using the jupyter notebook helps a lot to see the outputs are various stages.
- because we have different stage to see the things going on, like after importing the dataset, we need to see what's in it, what type of things we are using in it, the train and test split part, etc.


## The whole workflow in the model
- we need to know all the information gains of the attributes, the attribute having the highest info gain will be the root node, because it decides the decision tree.
- to calculate the info gain of the attribute, the first thing we need to calculate the entropy of the whole dataset, and the entropy of the each individual value of that attribute. for example: attribute=outlook, values of attribute are (sunny, overcast, rain), now we need to calculate the entropy of the whole dataset, entropy of the sunny, entropy of overcast and entropy of rain. then we will get info gain of the attribute=outlook. **see the decision_tree1.png image from the images folder of this repo at this point**
- ![photo](../images/decision_tree1.png)