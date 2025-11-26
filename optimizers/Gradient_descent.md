# What is it 
- *It's an iterative optimization algorithm which is used to train the machine learning models to adjust the parameters and minimize the loss function.*
- you can also say it like, its a **first order** optimization algorithm to adjust the parameters to get the minimum loss.
	- you'll later know why i used the term first order there.
# How it optimizes the loss !!
- mainly there are three types of this, and three of them work differently.
- it intially computes the gradients and later updates them eventually. it stop until it reaches a stopping condition which are maximum iteration, convergence or lack of imporovements etc.
## Types of gradient descent:
- stochastic:
- batch:
- mini batch: 
### How this works in the code:, NOte: this process used batch gradient descent. and this is just for example.
- you start with random parameters (e.g, slope m, intercept c) and at the end you should find the values which minimize the loss.
- at each step:
	- you compute the gradients (i.e, slope and loss function).
	- update the gradients. 
	- you repeat this until you get the loss as less as possible where the loss wont reduce anymore or till a stopping point.
- lets get an example of this:
	- objective:
		- minimize the loss function, for example the mean squared error (mse). the parameters would be m and c.
		- the model : y=mx+c
		- loss function(mse): 1/2*sum(y-y_pred)^2
	- computing the gradients: we use partial derivatives with respect to m and c to do this in the math, but lets assign the formula of that calculation to a variable, so it would be as follow:
		- m_gradient=(-2/n)*sum(x(y-y_pred))
		- c_gradient=(-2/n)*sum(y-y_pred)
	- now updating the gradients: we have the formula for that where we need learning rate to tell the model how much fast should it learn. generally the learning rate is initialized in the start,example 0.001 or 0.0001 or 0.00001.
		- m=m-learning rate*m_gradient
		- c=c-learning rate*c_gradient
