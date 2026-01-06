# Decision trees for regression
- When the target variable is continuous (e.g., predicting house prices or temperatures), decision trees are used for regression.
    - The standard algorithm here is CART (Regression mode).
    - Instead of entropy or Gini, CART uses variance reduction (or standard deviation reduction) to decide splits.
    - At each step, the algorithm chooses the feature that most reduces the variability of the target values in the child nodes compared to the parent node. This ensures that predictions at the leaves are as accurate as possible by minimizing prediction error.
- as you can see that the target attribute in the classification example (play tennis) has the only two values, yes or no, which is discrete. But here we are going to have real numbers 

### NOTE: all the images reagrding this explanation will be inside the DTR_images (Decision Tree Regression images) folder of images folder. so that i need not to mention.

## The whole worflow using a example 
- lets take a dataset which has 5 attributes (columns) 'outlook', 'temp', 'humidity', 'wind' and 'golf players' (target attribute).
- in the classification we took the information gain to make splits, but here we are taking **standard deviation reduction to make splits**. so the workflow goes as follow.
- first we need to calculate the standard deviation (SD) for the whole dataset (i.e. SD of target attribute), next the reduction in SD for each of the other individual attributes, based on that reduction we take the specific attribute and start building the tree. So, SD of the target attribute = 9.32 if you calculate.
- now after that we need to calculate the SD of individual attribute except the target attribute, late we will calculate the reduction in SD.
- same as in the classifcation as we did again for the values inside a attribute seperately, we do that same here. first outlook = sunny, outlook = overcast, etc, we now calculat the SD, weighted SD and lastly the reduction in SD. lets start with outlook = sunny **and the caculation is shown in image1.png**, after doing to all the other values you get the SD of each one and you calculate the weighted SD and next the reduction in SD, **all those calcuations are in one image naming image2.png.**
- now for temp, humidity, wind. all the reduction in SD in one, **image3.png**.
- you can now first tree, **image4.png**.
- now start with the sunny value and find its SD by only taking the rows which have the outlook as the sunny, we get 7.78 if calculated. now excluding the outlook attribute in from that table **image5.png**, start finding the SD of the other attributes. start with temp attribute now, we have 3 values, 'hot', 'cold', 'mind'. consider taking the rows with only temp=hot and find the SD. final table with all the SDs of the values, weighted SD, and reduction in SD,**image6.png**. 
- now all the SD of attribute exluding the outlook attribute, **image7.png**. 
- here comes very important part, first see the **image8.png**, you can see cool has only one example, mild and hot have 2 examples. what you can notice as we have only 1 example for outlook=sunny with temp=cool, as the values is 38 and the answer will be 38 directly, but it'll not be the case for outlook=hot and mild, we need to extend the tree from there, here's the tricky part, if we start extending the tree from there we get into problem naming **"overfitting", because we have only few examples, so we use a technique naming prunning, what we do in that technique is if we have less than 5 example we'll return the average of that examples as answer.**