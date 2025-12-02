# What is it 
- *It's an iterative optimization algorithm which is used to train the machine learning models to adjust the parameters and minimize the loss function.*
- you can also say it like, its a **first order** optimization algorithm to adjust the parameters to get the minimum loss.
	- you'll later know why i used the term first order there.
# How it optimizes the loss !!
- mainly there are three types of this, and three of them work differently.
- it intially computes the gradients and later updates them eventually. it stop until it reaches a stopping condition which are maximum iteration, convergence or lack of imporovements etc.
## Types of gradient descent:
1) stochastic: takes only single sample at a time to calculate the gradients.
2) batch: uses the whole dataset to calculate the gradients.
3) mini batch: you create batches and and for every set of batch you calculate the gradients and update them.
- you can see, how the three work differently below.
## Process to see how thery work in the code.
- I'll mention how all these work in the code seperately, but before konwing how they work, you need to knwow how the process starts and the key points and variables needed for that. if you see the code you'll get some idea about the structure.Now let's start
- the variables common for three of these are as follow:
	- learning rate: you give a small value for this and in the codes we give it a 0.1.
	- m and c: we intially give them both zeros.
	- n: length of the x.
	- itterations: we gave 1000.
- I want you to see the code of simple linear regression once to see from where the optimizer part starts. And i've aloso mentioned that part clearly.
- Now lets see how those work.
### 1) How **Batch gradient descent** works in the code.
