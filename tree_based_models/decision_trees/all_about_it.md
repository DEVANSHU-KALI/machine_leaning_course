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

## Advantages and Disadvantages
- Easy to interpret | Can over fit if too deep
- Handles categorical & numerical data | Sensitive to small data changes
- No assumptions about distribution | Biased toward features with many levels

## More About the Code
- From here understanding new concepts help a lot in the coding like, using datasets, data transformation, data visualization, evaluation metric, splitting data 

## Example 
- Lets take the dataset like this 

| ID | Age | Student | Buys |
|----|-----|---------|------|
| 1  | 22  | Yes     | Yes  |
| 2  | 25  | Yes     | Yes  |
| 3  | 47  | No      | No   |
| 4  | 52  | No      | No   |
| 5  | 46  | Yes     | Yes  |
| 6  | 56  | No      | No   |

- Now as there are three no's and three yes's the entropy is maximal
- Entropy and information gain (formulas):
    - Entropy of a set `S` with class proportions `pi` is:
      $$Entropy(S) = -\sum_{i} p_i \log_2 p_i$$
    - Information gain on splitting an attribute `A` is:
      $$IG(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$$
- These are standard definitions for choosing the splits.
- Root entropy.
    - Two classes with equal proportions: `P_yes = P_no = 0.5`
- Candidate split 1 - Student (Yes / No)
    - `student = yes`: IDs 1, 2, 5 -> All yes -> `entropy = 0`
    - `student = no`: IDs 3, 4, 6 -> All no -> `entropy = 0`
    - Weighted entropy:
        - `(3/6) * 0 + (3/6) * 0 = 0`......, thats because, we have 3 out 6 as yes and 3 out of 6 as no, so we got 3/6 two times and multiplied with zero is their entropies
        - So `IG(S, Student) = 1.0 - 0 = 1.0`
        - This is perfect split which yields the pure leaves.

## Points to Note:
- From this stage learning using the Jupyter Notebook helps a lot to see the outputs are various stages.
- Because we have different stage to see the things going on, like after importing the dataset, we need to see what's in it, what type of things we are using in it, the train and test split part, etc.