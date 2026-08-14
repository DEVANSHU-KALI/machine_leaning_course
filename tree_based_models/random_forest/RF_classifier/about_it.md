# What Is It?
- As the name suggests, we are using the random forest for classification problem.

### NOTE: Images reading this page will be in the **`images/RFC_images`** path.

## Process of Creating a Random Forest
- Lets take a dataset for example as shown in the **`image1.png`.**
- The first step is to create new datasets from the original dataset. For example lets create only 4, in real world we build more that 100s of datasets.
- We are going to select the rows randomly for each new dataset from the original dataset, and every new dataset will be the same size as the original dataset. 
- Instead of showing all the new dataset, Simple I'm showing a image which simply show the ids of the rows, To get a understanding on what's happening. See the **`image1.png`,** you can see the new subsets of data in the image from the original dataset. There are two things to know from the image:
    - You can see that the Ids are repeated in the new datasets, that's because of randomness, that's the reason why this is called random forest and we'll know more info about that in further points.
    - The process of creating new datasets like that is can **Bootstrapping**.
- Now we'll train a decision tree on each of the new dataset independently, but here's a twist, we wont use all the features of each dataset to train the decision tree, we randomly select subset of features for each tree and train on them, for example, for first new dataset, we take `x0` and `x1`, for next `x2` and `x3`, and so on. Optionally we see all the decision trees for the datasets. **`image2.png`**.
- Now we might wonder, how is the prediction done, for that lets take a new data point, we'll pass this data point through each tree one by one and note down the predictions. **`image3.png`**.
- As this is a classification problem we are going to take the majority votes. Which is 1 in this case as in the image shown. This process of combining results from multiple models is called **Aggregation**, so in this random forest, we first perform bootstrapping and then aggregation. **In general this is called bagging (bootstrapping + aggregation)**. 
- Now that's all how the random forest works.

## Important Points to Know About This Model
1) Why is this called random?
- That's because we used two random processes bootstrapping and random feature selection.
2) Why bootstrapping and random feature selection?
- Bootstrapping ensures that we are not using the same dataset for every tree, so in a way it helps the mode to be less sensitive to the original data, and the random feature selection helps to reduce the correlation between the trees. If we use every feature, then the most of the trees would have the same decision nodes and they will act very similarly, that will increase the variance. There is another use of the random feature selection: some of the trees will be trained on less important features, so they will give bad predictions, but there will also be some trees that give good predictions, so they will balance it out.
3) What is the ideal size of the feature subset?
- In our case we took 2 features which is close to the square root of the no.of features we have, which is 5. Researchers have found values close to the log or square root of the no.of features work well.
4) How to use this model for the regression problem?
- While combining the predictions just take the average and you can use it for the regression problem. And we'll discuss about it later.

## Important Points to Know
1) If you understand this model very clearly, you would've understand that the output of the model is based on the which class gets more votes (I mean the class which most of the decision trees return). But what if both the positive class and negative class have same number of votes? If we take 100 trees, and 50 trees return yes and 50 trees return no, for a new data point, what will the model predict?
- That's a insightful question to know. It's statistically very rare case that we get into this type of situations. Here's how the model handles those ties:
    1) Implementation-specific tie breaking:
    - Most libraries (like `sklearn` in Python) don't actually flip a coin, they actually follow a consistent internal logic.
        - Lowest index/Alphabetical order: Its simple, if both have the same number of votes the lowest one is declared as the output, in 1 and 0, 0 is declared as the output.
            - Now you might wonder, what if there are yes or no, true or false instead of 0 and 1. It doesn't matter because, in the implementation part you are going to transform those into binary values, and you get 0, which maybe no or false in most cases, and it also depends on what you declare, if you declare 0 as yes and 1 as no, the answer will be yes in this case. This is rare but maybe some people do this.
            - Actually the model is not caring about the 0 or 1, it's caring about the probability under the hood. If you have 100 trees and 50 vote "Yes" (1), the predicted probability `P` is: `P(class = 1) = summation(votes) / n_estimator = 50 / 100 = 0.5`
            - In Python, the `predict()` function uses default threshold of `>0.5` to return 1. Since 0.5 is not greater than 0.5, it returns 0.
    2) How to prevent ties:
    - Usually you can simply pick odd number of trees, like instead of taking 100, take 99 or 101. So the mathematical tie will be impossible.