# Decision trees for regression
- When the target variable is continuous (e.g., predicting house prices or temperatures), decision trees are used for regression.
    - The standard algorithm here is CART (Regression mode).
    - Instead of entropy or Gini, CART uses variance reduction (or standard deviation reduction) to decide splits.
    - At each step, the algorithm chooses the feature that most reduces the variability of the target values in the child nodes compared to the parent node. This ensures that predictions at the leaves are as accurate as possible by minimizing prediction error.
- as you can see that the target attribute in the classification example (play tennis) has the only two values, yes or no, which is discrete. But here we are going to have real numbers 

### NOTE: all the images reagrding this explanation will be inside the DTR_images (Decision Tree Regression images) folder of images folder. so that i need not to mention 

## The whole worflow using a example 
- lets take a dataset which has 5 attributes (columns) 'outlook', 'temp', 'humidity', 'wind' and 'golf players' (target attribute).
- in the classification we took the information gain to make splits, but here we are taking **standard deviation reduction to make splits**. so the workflow goes as follow.
- first we need to calculate the standard deviation (SD) for the whole dataset (i.e. SD of target attribute), next the reduction in SD for each of the other individual attributes, based on that reduction we take the specific attribute and start building the tree. So, SD of the target attribute = 9.32 if you calculate.
- now after that we need to calculate the SD of individual attribute except the target attribute, late we will calculate the reduction in SD.
- same as in the classifcation as we did again for the values inside a attribute seperately, we do that same here. first outlook = sunny, outlook = overcast, etc, we now calculat the SD, weighted SD and lastly the reduction in SD. lets start with outlook = sunny **and the caculation is shown in image1.png**, after doing to all the other values you get the SD of each one and you calculate the weighted SD and next the reduction in SD, **all those calcuations are in one image naming image2.png.**
- now for the 'temp' attribute, **image3.png.**, next for humidity, and wind. all the reduction in SD in one , **image4.png**.
- as we can see that outlook has the max reduction in SD, so we take that as root node. now same as we did in the classification part, we do the same here, exept the outlook attribute, consider the other attribtes and find which of them has the highest reduction in SD to make a new branch for the outlook.