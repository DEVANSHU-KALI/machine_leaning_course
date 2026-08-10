# Decision Trees as Classification
- When the target variable is categorical (e.g., “spam” vs “not spam”), decision trees are used for classification.
    - ID3 algorithm builds trees using Information Gain based on entropy.
    - C4.5 improves on ID3 by handling continuous attributes, missing values, and pruning, and it uses Gain Ratio as its splitting metric.
    - CART (Classification mode) is another algorithm that uses the GINI Index to measure impurity. In all cases, the goal is to choose splits that maximize class purity in the resulting subsets.
    - **We are using the `ID3` here.**

### NOTE: Images regarding the explanation will be inside the `DTC_images` (Decision Tree Classification images) folder inside `images` folder of the repo.

## The Whole Workflow in the Model Using a Example 
- Lets take a dataset with 5 attributes (columns) `'outlook'`, `'temp'`, `'humidity'`, `'wind'` and lastly `'play tennis'` (target attribute).
- We need to know all the information gains of the attributes, the attribute having the highest info gain will be the root node, because it decides the decision tree.
- To calculate the info gain of the attribute, the first thing we need to calculate the entropy of the whole dataset, and the entropy of the each individual value of that attribute. For example: `attribute = outlook`, values of attribute are (`sunny`, `overcast`, `rain`), now we need to calculate the entropy of the whole dataset, entropy of the `sunny`, entropy of `overcast` and entropy of `rain`. Then we will get info gain of the `attribute = outlook`. **See the `decision_tree1.png` image from the `DTC_images` folder at this point.**
- Now take all the info gains of the attribute at a place and see which is the highest, and take the attribute as the root node. **See the `decision_tree2.png` image from the `DTC_images` understanding this point.**
    - For example, lets take as you saw the image, at the end outlook has the highest info gain, now it'll be the root node and the leaf node of that root will be the three values as branches. **See that in the image naming `decision_tree3.png`.**
    - A point to note, we need to also note the target variables of the each branch. If you 