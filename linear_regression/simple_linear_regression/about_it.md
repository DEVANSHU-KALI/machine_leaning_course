Note: I hope you came here after reading about the linear regression.
# What does it mean
- simple linear regression only models **relationship between 1 dependent variable and 1 independent variable**.
- **Formula** for this model is: y_pred=m*x+c
- where the m and c are explained in the ___"All_about_linear_regression.md"___ page
- as explained that page, the next step would be the optimization step and you can see that clearly happening in the code.
- for more info about the otpimation step visit the ___gradient descent page from the optimizer folder___, there the process explained also works for this algorithm, i also mentiond this model and took this as example to explain that optimzer.
- you can simple learn how thats working and also get hands on the code while learning. 

## How to execute the code: 
- you need to install the numpy and matplotlib library.
- then copy the code to any of your ide or python interpreter and run it, you'll see the final parameters and a graph.
- Note: the code provided for this uses batch gradient desent optimizer, and this is a iterative apporach to find the best fit line for the given data points
- But you can also use the closed form solution (Normal Equation) to find the best fit line directly without iteration. The closed form solution is computationally more efficient for small datasets, but for larger datasets or when dealing with multiple features, gradient descent is often preferred due to its scalability and flexibility. to use this closed form solution, 
**you need to import the simple linear regression model from the sklearn.linear_models library** and do some tweeks to execute the code. After this, The code looks way smaller than this. 