Note: I hope you came here after reading about the linear regression.
- here this simple linear regression only models **relationship between 1 dependent variable and 1 independent variable**.
- other than that all the things are same, from calculating the gradients and updating them and plotting the graph.
- you need to install the numpy and matplotlib library.
- then copy the code to any of your ide or python interpreter and run it, you'll see the final parameters and a graph.
- Note: the code provided for this uses batch gradient desent algorithm, and this is a iterative apporach to find the best fit line for the given data points
- But you can also use the closed form solution (Normal Equation) to find the best fit line directly without iteration. The closed form solution is computationally more efficient for small datasets, but for larger datasets or when dealing with multiple features, gradient descent is often preferred due to its scalability and flexibility. to use this closed form solution, 
**you can simply import the linear regression model from the sklearn.linear_models library and run the code.** 