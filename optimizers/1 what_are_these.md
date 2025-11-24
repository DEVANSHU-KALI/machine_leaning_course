# What are optimizers 
- You can say them like algorithms which **reduce the loss by adjusting the model parameters through multiple iterations**.
- there many types of these which are used according to the situation.
# Why are these actually used ?
- These are used in the process to find the optimal parameters for better predictions.
# Can't we run the model without these?
- yes, but only in some special cases.
    - for some models like linear regression, there's closed form solution using the matrix algebra, in such cases you dont need any iterative approach, the best parameters are computed automatically.
- for most ml models (dl and large datasets), closed form dosent exist. so optimizer come into play.
- these optimizers iteratively adjust the parameters to reduce the loss as in the first point.
## Real world example:
-  lets take a situation like, you and your friend wrote a test which is for 50 marks and you got the results, you know you marks but you asked your friend, how he wrote his exam and about his marks, he said he wrote the exam well and said to guess the marks, now here's the main part.
- lets take like, you guessed 35, and he said nope!, you guessed low, you again think and say like 40, now he said 'you are close', finally you said 45 and he said, 'yes i got 45'.
- what actually you should notice is, how iteratively you changed the marks so that, you were far at first then got close and then at the end you got the answer. 
### What you should understand from the example
- all the optimizers do the same, they go itratively, actually in the real world senario in ML its not that easy to get that perfect, you get close. 