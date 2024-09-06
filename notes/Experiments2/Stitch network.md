In this set of experiments we try to understand whether or not a network can learn to use a fragment. For this, we use a base network and a pre-trainer fragment. Then we connect intermediate layers of the base network to intermediate layer of the fragment, through what we're going to call stitching layers. We will also freeze the fragment (but, obviously, not the stitching layer). 

In specific we are going to use the same function we used in experiments1:
$$\begin{equation}
f(x1​,x2​)=x_1 x_2​+\sin(x_1​)−\cos(x_2​)+x_1^2​−x_2^2
\end{equation}$$​For the fragments we are going to use one of the operators that is obviously necessary such as sine or cosine. We might also test what happens if we have a useless fragment just to compare what happens.


### Experiment 2.1. - Approximate function with useful sin fragment
see [[2.1.  - Approximate function with useful sin fragment]]


### Experiment 2.2. - Approximate function with useless fragment
see [[2.2. - Approximate function with useless fragment]]



###  Appendix

An interesting thing happened when I was trying to train a neural network to approximate the sin function. I created a validation set that extended beyond the domain of the training set. The neural network with ReLUs was awful at generalizing the sin function. It didn't seem to learn the repetitive behavior of it. I then changed the activation function to sigmoid and it improved tremendously. It went from 2.5 to 0.46 MAE. For fun, I tried using torch.sin as the activation function, even though this is not an activation function. 

Here's an idea: using/investigate about cyclical activation functions.

This is a motivation for our work. Networks today use ReLUs or GeLUs, which means that if they need to learn Sinusoidal functions, they won't be able to. But, if we have these functions available for them to use, it might help. This is just an example of what can come from our work. 


