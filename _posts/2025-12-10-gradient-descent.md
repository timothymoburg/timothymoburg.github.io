# Gradient Descent Algorithms
## Introduction
	
The high-level goal of many machine learning methods is to minimize a real-valued objective function (also called a cost function or loss function) $F: \mathbb{R}^n \rightarrow \mathbb{R}$ with respect to its $n$ parameters $\mathbf{\theta}_t = \left( \theta_1, \dots , \theta_n \right)$. One successful strategy of doing so, if $F$ is differentiable, is to use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), whereby we [iteratively](https://en.wikipedia.org/wiki/Iterative_method) update the parameters in the direction of steepest decrease, that given by the negative gradient of $F$, and gradually descend on a local minimum. This method is equivalent to a person lost in some mountains at night and trying to get to the base employing the strategy of repeatedly taking steps downhill in the direction where the terrain is the steepest at each spot. Therefore, if we denote $\mathbf{\theta}_0$ to be our starting location and $\boldsymbol{\theta}_t$ be our location after $t$ steps (and our $t^{th}$ approximation of the minimizer), then our next location is given by:

$$
	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \eta \mathbf{g}_{t},
$$

where $\boldsymbol{\theta}_{t}$ is our previous step, $\eta$ is a tunable parameter that determines the step size (also called the learning rate), and $\mathbf{g}_{t} = \nabla F\left(\boldsymbol{\theta}_{t}\right)$ is the Euclidean gradient of our objective function with respect to our parameters at time $t$.

Although gradient descent is a clever method for trying to find minimums, it does have some issues. The contours of many objective functions are far from ideal and can have sections where the direction of the gradient can fluctuate drastically in an area smaller than the step size taken, which can lead to chaotic routes towards the bottom or even prevention of the method from finding a minimum. To address these concerns, various modifications to gradient descent have been proposed. As of 2025, all but one ([L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) of the most popular of these methods are first-order ones, not utilizing second derivative information due to the computational cost.

In all of these methods, we will denote $\mathbf{\theta}_t$ to be the parameters of our objective function at time $t$, we will denote $\mathbf{g}_t$ to be the Euclidean gradient of our objective function with respect to our parameters at time $t$, we will denote $\circ$ to be the Hadamard product, and allow for broadcasting.


