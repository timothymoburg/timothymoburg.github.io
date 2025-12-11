## Introduction
	
The high-level goal of many machine learning methods is to minimize a real-valued objective function (also called a cost function or loss function) $F: \mathbb{R}^n \rightarrow \mathbb{R}$ with respect to its $n$ parameters $\mathbf{\theta}_t = \left( \theta_1, \dots , \theta_n \right).$ One successful strategy of doing so, if $F$ is differentiable, is to use [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), whereby we [iteratively](https://en.wikipedia.org/wiki/Iterative_method) update the parameters in the direction of steepest decrease, that given by the negative gradient of $F$, and gradually descend on a local minimum. This method is equivalent to a person lost in some mountains at night and trying to get to the base employing the strategy of repeatedly taking steps downhill in the direction where the terrain is the steepest at each spot. Therefore, if we denote $\boldsymbol{\theta}_0$ to be our starting location and $\boldsymbol{\theta}_t$ be our location after $t$ steps (and our $t^{th}$ approximation of the minimizer), then our next location is given by:

$$
	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \eta \mathbf{g}_{t}
$$

where $\boldsymbol{\theta}\_{t}$ is our previous step, $\eta$ is a tunable parameter that determines the step size (also called the learning rate), and $\mathbf{g}\_{t} = \nabla F\left(\boldsymbol{\theta}\_{t}\right)$ is the Euclidean gradient of our objective function with respect to our parameters at time $t$.

Although gradient descent is a clever method for trying to find minimums, it does have some issues. The contours of many objective functions are far from ideal and can have sections where the direction of the gradient can fluctuate drastically in an area smaller than the step size taken, which can lead to chaotic routes towards the bottom or even prevention of the method from finding a minimum. To address these concerns, various modifications to gradient descent have been proposed. As of 2025, all but one ([L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) of the most popular of these methods are first-order ones, not utilizing second derivative information due to the computational cost.

In all of these methods, we will denote $\mathbf{\theta}\_{t}$ to be the parameters of our objective function at time $t$, we will denote $\mathbf{g}\_t$ to be the Euclidean gradient of our objective function with respect to our parameters at time $t$, we will denote $\circ$ to be the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)), and allow for broadcasting.

## Momentum
In gradient descent with momentum (also called the heavy ball method), we envision our parameter vector rolling down an error surface like a ball rolling down a hill with friction under the influence of gravity. Such a method yields an update rule that looks like  

$$
	\begin{align*}
		\color{red}\mathbf{v}_{t+1}\color{black} &= \gamma \mathbf{v}_{t} - \eta \mathbf{g}_{t}\\
		\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_{t} + \color{red}\mathbf{v}_{t+1}\color{black}
	\end{align*}
$$  

where $\gamma$ is a parameter similar to a coefficient of friction and $\eta$ is the global learning rate. Continuing with the dynamical ball analogy for this update rule, we note that, using the definition of the variable $\mathbf{g}_t$, combining the two lines and rearranging yields  

$$
	\begin{align*}
		-\eta \nabla F\left(\boldsymbol{\theta}_{t}\right) &= -\eta \mathbf{g}_{t}\\
		&= \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_{t} - \gamma \left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right)\\
		&= \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_{t} + \left[-\left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right) + \left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right)\right] - \gamma \left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right)\\
		&= \boldsymbol{\theta}_{t+1} - 2\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1} + \left(1 - \gamma\right) \left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right)\\
		-\nabla F\left(\boldsymbol{\theta}_{t}\right) &= \frac{\boldsymbol{\theta}_{t+1} - 2\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}}{\eta} + \frac{1 - \gamma}{\eta} \left(\boldsymbol{\theta}_{t} - \boldsymbol{\theta}_{t-1}\right).
	\end{align*}
$$  

The first term on the right is a <a href="https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences">finite difference approximation</a> for the second derivative $\boldsymbol{\theta}^{\prime\prime}$, while the second term on the right is a scaled <a href="https://en.wikipedia.org/wiki/Finite_difference#Relation_with_derivatives">finite difference approximation</a> for the first derivative $\boldsymbol{\theta}^{\prime}$. Therefore, the momentum update rule corresponds with a second order ordinary differential equation, similar to the equations of motion for a particle of unit mass being acted on by a conservative force (like gravity) and a frictional force (like air resistance) with coefficient $\frac{1 - \gamma}{\eta}$.  

There are pros and cons to gradient descent with momentum. Since velocity can sort of be thought of as a source of memory due to it being the result of all the forces that have acted on it in the past, the vector can naturally grow in magnitude in the direction of a minimum while being somewhat resistant to drastic disruptions. This is sometimes a good quality to have, but can be a bad quality if it delays the response in adjusting to a better direction, such as when accumulated momentum at the bottom of a surface is high and causes the method to overshoot the minimum. A variant of this momentum, called Nesterov Accelerated Gradient slightly improves on this deficiency by performing the gradient calculation not at the given spot but at the look-ahead spot that a step in the direction of momentum would yield.  

## AdaGrad
AdaGrad, a portmanteau of <ins>ada</ins>ptive <ins>grad</ins>ient, is a method with an adaptive learning rate that for each parameter gets adjusted based on the magnitudes of its historic gradients. Such a method yields an update rule that looks like  

$$
\begin{align*}
	\color{blue}\mathbf{r}_{t}\color{black} &= \mathbf{r}_{t-1} + \mathbf{g}_{t} \circ \mathbf{g}_{t}\\
	\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_{t} - \frac{\eta}{\epsilon + \sqrt{\color{blue}\mathbf{r}_{t}\color{black}}}\circ \mathbf{g}_{t}
\end{align*}
$$  

where $\eta$ is the global learning rate, $\epsilon$ is some small number to prevent division by zero, and $\mathbf{r}_0 = \mathbf{0}$. This update rule has the effect that large historic gradients will increase the denominator in the second line which leads to smaller effective learning rates, while small historic gradients will decrease the denominator in the second line which leads to larger effective learning rates. One main issue with this property is that it can lead to premature stopping since there is a continual accumulation of squared gradients throughout training. Despite this shortcoming, AdaGrad does perform well with sparse gradients.  

## RMSProp
RMSProp, another method with an adaptive learning rate, can be thought of as windowed AdaGrad. Whereas AdaGrad keeps a complete history of historic squared gradients, RMSProp keeps an [exponential moving average](https://en.wikipedia.org/wiki/Exponential_smoothing) of historic squared gradients. Iterations with the form of the first line of the following update rule are called an exponential moving average, and they are designed such that the weights fall exponentially the further back one goes into the time series. So for the first line of the following update rule, if we expand the previous iterate, $\mathbf{r}\_{t-1}$, then it follows that $\mathbf{r}\_t = \gamma (\gamma \mathbf{r}\_{t-2} + (1-\gamma)\mathbf{g}\_t \circ \mathbf{g}\_t) + (1-\gamma)\mathbf{g}\_t \circ \mathbf{g}\_t$. Continuing this process recursively and defining $\mathbf{r}\_0 = \mathbf{0}$, we see that $\mathbf{r}\_t = \sum_{k=1}^{t} \gamma^{t-k} (1-\gamma) \mathbf{g}\_{k} \circ \mathbf{g}\_{k}$, which can also be interpreted as a [convolution](https://en.wikipedia.org/wiki/Convolution") of the filter $h_k = \gamma^{t-k} (1-\gamma)$ with the time series data. So the effect of an exponential moving average is that older observations are given exponentially smaller weights and this rate of decay is governed by the size of $\gamma$, where $0<\gamma<1$. This modification to AdaGrad, provides RMSProp with the update rule  

$$
	\begin{align*}
		\color{blue}\mathbf{r}_{t}\color{black} &= \gamma \mathbf{r}_{t-1} + (1-\gamma)\mathbf{g}_{t} \circ \mathbf{g}_{t}\\
		\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_{t} - \frac{\eta}{\epsilon + \sqrt{\color{blue}\mathbf{r}_{t}\color{black}}}\circ \mathbf{g}_{t}
	\end{align*}
$$  

where $\eta$ is the global learning rate, $\epsilon$ is some small number to prevent division by zero, and $\mathbf{r}_0 = \mathbf{0}$. RMSProp tends to perform well in on-line and non-stationary settings.  

## Adam
Adam, a portmanteau of <ins>ada</ins>ptive <ins>m</ins>oment estimation, is another method with an adaptive learning rate, especially popular in the deep learning community where it was the optimizer chosen to train OpenAI's ChatGPT. Adam can be loosely thought of as RMSProp with windowed momentum. So, like in RMSProp, we will use an exponential moving average of squared gradients given here by $\color{black}\mathbf{v}\_{t} = \beta\_{2} \mathbf{v}\_{t-1} + \(1-\beta\_{2}\)\mathbf{g}\_{t} \circ \mathbf{g}\_{t}$, but also incorporate an exponential moving average of momentum given here by $\color{black}\mathbf{m}\_{t}\color{black} = \beta\_{1} \mathbf{m}\_{t-1} + \(1-\beta\_{1}\)\mathbf{g}\_{t}$.
 Such properties yield a method that is almost Adam but not quite it, sometimes referred to as Adam (biased). Adam (biased) has the update rule  

$$
	\begin{align*}
		\color{red}\mathbf{m}_{t}\color{black} &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_{t}\\
		\color{blue}\mathbf{v}_{t}\color{black} &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_{t} \circ \mathbf{g}_{t}\\
		\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_{t} - \frac{\eta}{\epsilon + \sqrt{\color{blue}\mathbf{v}_{t}\color{black}}}\circ \color{red}\mathbf{m}_{t}\color{black}
	\end{align*}
$$  

where $\eta$ is the global learning rate, $\epsilon$ is some small number to prevent division by zero, $\beta\_{1},\beta\_{2} \in [0,1)$ are decay rates, and $\mathbf{m}\_{0} = \mathbf{v}\_{0} = \mathbf{0}$.  

Unfortunately, initializing $\mathbf{m}\_{t}$ and $\mathbf{v}\_{t}$ to be the zero vector at time $t=0$, does bias these variables towards zero. To correct for this bias in $\mathbf{m}\_{t}$ (the calculation equivalent for $\mathbf{v}\_{t}$), let us see how the [expectation](https://en.wikipedia.org/wiki/Expected_value) differs between $\mathbf{m}\_{t}$ and $\mathbf{g}\_{t}$. Letting $\mathbf{g}\_{1},\cdots,\mathbf{g}\_{T}$ be the [independent and identically distributed](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) gradients at subsequent timesteps, if we take the expectation of both sides of $\mathbf{m}\_{t}$, then use the linearity property of expectations, and finally recognize the evident geometric series, it follows that  

$$
\begin{align*}
    \mathbb{E}[\mathbf{m}_t] &= \mathbb{E}[\beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t] \\
    &= \mathbb{E}\left[\sum_{k=1}^{t} \beta_1^{t-k} (1-\beta_1) \mathbf{g}_k \right] \\
    &= \mathbb{E}[\mathbf{g}_t] (1-\beta_1) \sum_{k=1}^{t} \beta_1^{t-k} + \zeta \\
    &= \mathbb{E}[\mathbf{g}_t] (1-\beta_1) \frac{1-\beta_1^{t}}{1-\beta_1}  + \zeta \\
    &= \mathbb{E}[\mathbf{g}_t] (1-\beta_1^{t}) + \zeta
\end{align*}
$$
  

where $\zeta = 0$ if $\mathbb{E}[\mathbf{g}\_{k}]$ is stationary. Therefore, dividing $\mathbf{m}\_{t}$ by $(1-\beta\_{1}^{t})$ corrects its initialization bias.  

Combining RMSProp with windowed momentum and bias correction yields the actual Adam update rule  

$$
	\begin{align*}
		\color{red}\mathbf{m}_{t}\color{black} &= \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_{t}\\
		\color{blue}\mathbf{v}_{t}\color{black} &= \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_{t} \circ \mathbf{g}_{t}\\
		\color{purple}\hat{\mathbf{m}}_t\color{black} &= \frac{\color{red}\mathbf{m}_t\color{black}}{1-\beta_1^t}\\
		\color{teal}\hat{\mathbf{v}}_t\color{black} &= \frac{\color{blue}\mathbf{v}_t\color{black}}{1-\beta_2^t}\\
		\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_{t} - \frac{\eta}{\epsilon + \sqrt{\color{teal}\hat{\mathbf{v}}_{t}\color{black}}}\circ \color{purple}\hat{\mathbf{m}}_{t}\color{black}
	\end{align*}
$$  

where $\eta$ is the global learning rate, $\epsilon$ is some small number to prevent division by zero, $\beta\_{1},\beta\_{2} \in [0,1)$ are decay rates, and $\mathbf{m}\_{0} = \mathbf{v}\_{0} = \mathbf{0}$.  

One advantageous property of Adam stems from probability theory. The $k^{th}$ moment of a random variable, $X$, is defined to be the expectation of the random variable to the $k^{th}$ power, $\mathbb{E}[X^{k}]$. So the expectation of the gradient is called the first moment (also known as the mean) of the gradient and, by the [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers), Adam's $\hat{\mathbf{m}}_t$ variable then is an estimate of it. Likewise, the expectation of the squared gradient is called the second moment (also known as the uncentered variance) of the gradient and, by the law of large numbers, Adam's $\hat{\mathbf{v}}_t$ variable is then an estimate of it. The ratio $\frac{\hat{\mathbf{m}}_t}{\hat{\mathbf{v}}_t}$ in Adam's effective stepsize can therefore be loosely thought of as a ratio of the mean to the square root of the variance (also known as the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)), which is sometimes referred to as the [signal-to-noise ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) (SNR). A higher signal-to-noise ratio here indicates that the expected gradient is much stronger than the uncertainty in the gradient estimates, signifying that the estimates are more reliable, and Adam, as desired, will take a larger effective stepsize. Whereas a lower signal-to-noise ratio here indicates that the expected gradient is much weaker than the uncertainty in the gradient estimates, signifying that the estimates are less reliable, and Adam, as desired, will take a smaller effective stepsize. Such a desirable property might partly explain its numerable successes in practice.  
