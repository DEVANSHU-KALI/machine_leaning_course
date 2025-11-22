The first thing, **what is this gradient descent actually**: *It's an iterative optimization algorithm which is used to train the machine learning models to adjust the parameters and minimize the loss function.*

## How actually its done in, lets get the core idea:
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

## types of gradient descent
- mostly we use the basic types in the beghining and go to only some of the advanced types when you enter the deeper levels of ML, but i've mentioned all the types ive got so far to let know that there are more types excluding the beghining level types.
# basic types of gradient descents:
1) batch gradient descent:
2) stochastice gradiente descent:
3) mini-batch gradient descent:
# advance types of gradient descents:
1) momementum:
2) nestevor accelerated gradient(NAG):
3) adagrad (adaptive gradient):
4) rmsprop (root mean square propogation):
5) adam (adaptive moement estimation):
6) ada max:
7) nadam (nesterov-accelerated adam):
# specialized / cutting-edge techniques:
1) AMSgrad:
2) LAMB (layer-vice adaptive momemtens optimizer for batch training):
3) lookadhead optimizer:
4) gradient clipping: