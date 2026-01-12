# What is it
- as the name suggests, we are using the random forest for classification problem.

### NOTE: images rearding this page will be in the **images/RFC_images** path.

## Process of creating a random forest
- Lets take a dataset for example as shown in the **image1.png.**
- The first step is to create new datasets from the original dataset. For example lets create only 4, in real world we build more that 100s of datasets.
- We are going to select the rows randomly for each new dataset from the original dataset, and every new dataset will be the same size as the original dataset. 
- Instead of showing all the new dataset, Simple I'm showing a image which simply show the ids of the rows, To get a understanding on what's happening. See the **image1.png,**, you can see the new subsets of data in the image from the original dataset. There are two things to know from the image
    - You can see that the Ids are repeated in the new datasets, that's because of randomness, that's the reason why this is called random forest and we'll know more info about that in further points.
    - the process of creating new datasets like that is can **Bootstraping**.
- Now we'll train a decision tree on each of the new dataset independently, but here's a twist, we wont use all the features of each dataset to train the decision tree, we randomly select subset of features for each tree and train on them, for example, for first new dataset, we take x0 and x1, for next x2 and x3, and so on. optionally we see all the decison trees for the datasets. **image2.png**.
- Now we might wonder, how is the prediction done, for that lets take a new data point, we'll pass this data point through each tree one by one and note down the predictions. **image3.png**.
- as this is a classification problem we are going to take the majority votes. which is 1 in this case as in the image shown. ssddc
