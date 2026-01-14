# What is it 
- *It's an iterative optimization algorithm which is used to train the machine learning models to adjust the parameters and minimize the loss function.*
- you can also say it like, its a **first order** optimization algorithm to adjust the parameters to get the minimum loss.
	- you'll later know why i used the term first order there.
# How it optimizes the loss !!
- mainly there are three types of this, and three of them work differently.
- it initially computes the gradients and later updates them eventually. it stop until it reaches a stopping condition which are maximum iteration, convergence or lack of improvements etc.
## Types of gradient descent:
1) stochastic: takes only single sample at a time to calculate the gradients.
2) batch: uses the whole dataset to calculate the gradients.
3) mini batch: you create batches and and for every set of batch you calculate the gradients and update them.
- you can see, how the three work differently below.
## Process to see how they work in the code.
- I'll mention how all these work in the code separately, but before knowing how they work, you need to know how the process starts and the key points and variables needed for that. if you see the code you'll get some idea about the structure. Now let's start
- the variables common for three of these are as follow:
	- learning rate: you give a small value for this and in the codes we give it a 0.1.
	- m and c: we initially give them both zeros.
	- n: length of the x.
	- iterations: we gave 1000.
- I want you to see the code of simple linear regression once to see from where the optimizer part starts. And i've also mentioned that part clearly.
- Now lets see how those work.
- **Note** : to direct implemnent these into the code, remove the lines explaining the process and directly replace the gradient calculation part with these, the other part is same for all.
### 1) How **Batch gradient descent** works in the code.
- the above mentioned vairables remain same. 
- you run a loop for itterations.
		- Y_pred = m * X + c
		- m_gradient = (-2/n) * np.sum(X * (Y - Y_pred))
		- c_gradient = (-2/n) * np.sum(Y - Y_pred)
		- note: the np.sum indicates that we are taking the whole dataset for each itteration to calculate the gradients and update them.
	- now the updating part
		- m = m - learning_rate * m_gradient
		- c = c - learning_rate * c_gradient

### 2) how **Stochastic gradient descent** works in the code
- the variables remains the same.
- loop through for itterations.
	- to take only one random sample at a time we do the above process
		- idx=np.random.randint(0,n) #where the n is length of n, as we know
		- X_i = X[idx]
		- Y_i = Y[idx]
		- Y_pred = m * X_i + c # intialize the formula, we use X_i because we created the new value for that
		- now the gradient caculating part
		- m_gradient = -2 * np.sum(X_i * (Y_i - Y_pred ))
		- c_gradient = -2 * np.sum(Y_i - Y_pred)
		- gradient updatation part
		- m = m - learning_rate * m_gradient
		- c = c - learning_rate * c_gradient

### 3) how **Mini-Batch gradient descent** works in the code
- again the variables remain the same.
- we intialize another variable batch_size=2, intialize this where we intialized the learning rate, m and c, you can also intialize this in between but, its a choice.
- batch_size=2 : that means we are taking 2 data points per batch.
- replace=True : this parameter, allows randomness in the data, which adds noise that helps convergence and generalize. randomness in sampling ensures the model always dosen't always see the data in the same order. It improves generalization by not overfitting to a fixed sequence of data. this is better for small datasets.s
- loop through for the itterations.
		- idx=np.random.choice(n,batch_size,replace=True)
		- X_batch=X[idx]
		- Y_batch=Y[idx]
		- Y_pred= m * X_batch + c 
		- calculating the gradients
		- m_gradient = (-2/X_batch) * np.sum(X_batch * (Y_batch = Y_pred))
		- c_gradient = (-2/X_batch) * np.sum(Y_batch - Y_pred)
		- gradient updatations part
		- m = m - learning_rate * m_gradient
		- c = c - learnign_rate * c_gradient