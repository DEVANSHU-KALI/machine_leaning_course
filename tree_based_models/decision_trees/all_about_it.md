# What is it?
- Decision trees are a machine learning concept used for predictive modeling. They represent decisions in a tree-like structure, where each internal node corresponds to a feature-based test, each branch represents the outcome of that test, and each leaf node gives a final prediction. The idea is to recursively split the dataset into subsets until the data in each subset is as “pure” or homogeneous as possible with respect to the target variable.
- we can use this concept for classification and regression.

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
- now take all the info gains of the attribute at a place and see which is the highest, and take the attribute as the root node. **see the decision_tree2.png image for getting the point**
    - for example lets take as you saw the image, at the end outlook has the highest info gain, now it'll be the root node and the leaf node of that root will be the three values as branches,**you can see that in the image naming decision_tree3.png** in the images folder.
    - a point to note, we need to also note the target variables of the each branch, if you notice the table, you will know the for the branch overcast all the target  variables are yes, **which means if the outlook is overcast the _chance of playing tennis (which is the actual target caloumn(you can see that in the table))_ will be yes for sure without any doubt.** but if there is a mix of yes and no, it should be decided using some math internally, **you can see that in the image naming decision_tree3.png** in the images folder.
- now as there are two node which should be decide as yes or no, we need to claculate them by taking the number of valuse that branch had in that table, like sunny had total 5 attributes (2 as yes and 3 as no), and rain has 5 attribute (3 as yes and 2 as no), we to consider those atttributes and do that calculation. now lets take the sunny as first, lets consider that 5 attributes of that sunny branch, and no this time as we took the outlook first, we dont take it again, now we that the remaining attributes temp, humidity and wind, and calculate the same info gain of the three attribute there and do the same procees as we did here. y**ou can see decision_tree4.png to see that**, you can see there humidity has more info gain. now observe the values of the humidity in the table, you can see that if the humidity is high we have NO and if normal it's YES, now simply the sunny branch should have a decision as humidiy attribute and if the humidiy is high, the answe will be no and if normal the answer would be yes. **see the decision_tree5.png**
- take for the rain branch, wind attribute will have the highest info gain, and the values are weak and strong, if we observe, we can clearly see, if wind is weak the answer will be yes, and if it's strong the answer will be no. the tree is completed. NOTE: im not mentioning the info gains of the attributes here, beacuse we alrealy understood the process, im showing the final tree now.
- **see the decision_tree6.png for the final tree**