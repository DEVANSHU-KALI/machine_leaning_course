# What is it
- as the name suggests, we are using the random forest for classification problem.

### NOTE: images reading this page will be in the **images/RFC_images** path.

## Process of creating a random forest
- Lets take a dataset for example as shown in the **image1.png.**
- The first step is to create new datasets from the original dataset. For example lets create only 4, in real world we build more that 100s of datasets.
- We are going to select the rows randomly for each new dataset from the original dataset, and every new dataset will be the same size as the original dataset. 
- Instead of showing all the new dataset, Simple I'm showing a image which simply show the ids of the rows, To get a understanding on what's happening. See the **image1.png,**, you can see the new subsets of data in the image from the original dataset. There are two things to know from the image
    - You can see that the Ids are repeated in the new datasets, that's because of randomness, that's the reason why this is called random forest and we'll know more info about that in further points.
    - the process of creating new datasets like that is can **Bootstrapping**.
- Now we'll train a decision tree on each of the new dataset independently, but here's a twist, we wont use all the features of each dataset to train the decision tree, we randomly select subset of features for each tree and train on them, for example, for first new dataset, we take x0 and x1, for next x2 and x3, and so on. optionally we see all the decision trees for the datasets. **image2.png**.
- Now we might wonder, how is the prediction done, for that lets take a new data point, we'll pass this data point through each tree one by one and note down the predictions. **image3.png**.
- as this is a classification problem we are going to take the majority votes. which is 1 in this case as in the image shown. this process of combining results from multiple models is called **Aggregation**, so in this random forest, we first perform bootstrapping and then aggregation, i**n general this is called bagging (bootstrapping + aggregation)**. 
- now that's all how the random forest works.

## important points to know about this model
1) why is this called random?
- that's because we used two random processes bootstrapping and random feature selection 
2) why bootstrapping and random feature selection?
- bootstrapping ensures that we are not using the same dataset for every tree, so in a way it helps the mode to be less sensitive to the original data, and the random feature selection helps to reduce the correlation between the trees, if we use every feature, then the most of the trees would have the same decision nodes and they will act very similarly, that will increase the variance. there is another use of the random feature selection, some of the trees will be trained on less important features, so they will give bad predictions, bu there will also be some trees give good predictions, so they will balance it out.
3) what is the ideal size of the feature subset?
- in our case we took 2 features which is close to the square root of the no.of features we have, which is 5. researchers have found values close to the log or square root of the no.of features work well
4) how to use this model for the regression problem?
- while combining the predictions just take the average and you can use it for the regression problem. and we'll discuss about it later.

## important points to know
1)  if you understand this model very clearly, you would've understand that the output of the model is based on the which class gets more votes (I mean the class which most of the decision trees return). but what if both the positive class and negative class have same number of votes?, if we take 100 trees, and 50 trees return yes and 50 trees return no, for a new data point, what will the model predict?
- that's a insightful question to know, It's statistically very rare case that we get into this type of situations. here's how the model handles those ties:
    1) implementation specific tie breaking:
    - most libraries (like sklearn in python) don't actually flip a coin, they actually follow a consistent internal logic.
    - 