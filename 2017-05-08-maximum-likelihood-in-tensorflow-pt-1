---
layout: post
title: 'Maximum likelihood in TensorFlow pt. 1'
---

Here are step-by-step examples demonstrating how to use TensorFlow's autodifferentiation toolbox for maximum likelihood estimation.  I show how to compute the MLEs of a univariate Gaussian using TensorFlow-provided gradient descent optimizers or by passing scipy's BFGS optimizer to the TensorFlow computation graph. 

- [MLE of univariate Gaussian with gradient descent](#mle-of-univariate-gaussian-with-gradient-descent)
	- [Preprocessing the data](#preprocessing-the-data)
	- [Part 1: Define computational graph](#part1)
	- [Part 2:  Run optimization scheme](#part2)
	- [Results](#results)
- [MLE of univariate Gaussian with Newton methods](#mle-of-univariate-gaussian-with-newton-methods)
	- [Computing the Hessian](#computing-the-hessian)
	- [Convexity of natural parameterization](#convexity-of-natural-parameterization)
	- [Using scipy's BFGS optimizer](#using-scipys-bfgs-optimizer)
- [Appendix](#appendix)
	- [Verify gradient and Hessian computations](#verify-gradient-and-hessian-computations)
	- [Gradient descent with natural parameters](#gradient-descent-with-natural-parameters)

The full code for each of these examples can be found in [this repo](https://github.com/kyleclo/tensorflow-mle).  The code snippets in this post are simplified for illustrative purposes, while the full code in the linked repo are executable.

## MLE of univariate Gaussian with gradient descent

Let's start with a simple example using gradient descent to find the MLE of a univariate Gaussian variable $X$ with mean $\mu$ and standard deviation $\sigma$.  We observe an iid sample of size $n = 100$.

```python
import numpy as np

TRUE_MU = 10.0
TRUE_SIGMA = 5.0
SAMPLE_SIZE = 100

np.random.seed(0)
x_obs = np.random.normal(loc=TRUE_MU, scale=TRUE_SIGMA, size=SAMPLE_SIZE)
```

We know that the correct MLEs are the sample mean and sample standard deviation of the data.  We will be evaluating whether TensorFlow's optimizer converges to these values.

#### Preprocessing the data

Before doing any optimization, let's standardize the raw data.

```python
CENTER = x_obs.min()
SCALE = x_obs.max() - x_obs.min()
x_obs = (x_obs - CENTER) / SCALE
```

Standardizing variables can make it easier to set reasonable defaults for initial values and learning rate.  Usually, `CENTER = x_obs.mean()` and `SCALE = x_obs.std()`.  For our example, we instead standardize the data using its min and max values just so things aren't too easy.

Since the transformation is linear and the parameters of interest are the mean and standard deviation, the MLEs computed from the standardized data can be transformed back to get the MLEs of the original data:

$$\widehat{\mu}_{original} = SCALE \cdot \widehat{\mu}_{standard} + CENTER$$

$$\widehat{\sigma}_{original} = SCALE \cdot \widehat{\sigma}_{standard}$$

Note that we won't always know how a specific transformation applied to the data will impact the parameters of interest.  

<a id="part1"></a>

#### Part 1:  Define computational graph

```python
import tensorflow as tf

# data
x = tf.placeholder(dtype=tf.float32)

INIT_MU_PARAMS = {'loc': 0.0, 'scale': 0.1}
INIT_PHI_PARAMS = {'loc': 1.0, 'scale': 0.1}
RANDOM_SEED = 0

# params
np.random.seed(RANDOM_SEED)
mu = tf.Variable(initial_value=np.random.normal(**INIT_MU_PARAMS),
                 dtype=tf.float32)
phi = tf.Variable(initial_value=np.random.normal(**INIT_PHI_PARAMS),
                  dtype=tf.float32)
sigma = tf.square(phi)

# loss
gaussian_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
log_prob = gaussian_dist.log_prob(value=x)
neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)

# gradient
grad = tf.gradients(neg_log_likelihood, [mu, phi])
```

Several things to note:

- The `tf.contrib.distributions` module provides implementations of common distributions, but its use is optional.  You can always simply define the loss function from scratch.

- TensorFlow doesn't seem to provide a way to explicitly enforce variable constraints, i.e. $\sigma \gt 0$.  To handle this, we use the parameterization $\sigma = \phi^2$: 

	- This means our loss function will be optimized with respect to $\phi \in \mathbb{R}$, and we won't have awkward gradient steps that push $\sigma$ outside the range of viable values.  

	- But this also means the solution is non-unique: Two values of $\phi$ correspond to a single optimal $\sigma$.  The loss function is symmetric around $\phi = 0$, and we might be concerned with potential jumping between solutions if the true $\sigma$ is very small (causing the optimal $\phi$ values to be sufficiently close to each other in parameter space).  

	- For more discussion on this, see the section about variance-covariance parameterization using the Cholesky decomposition in [Pinheiro, J. C., & Bates, D. M. (1996)](https://pdfs.semanticscholar.org/2ff5/5b99d6d94d331670719bb1df1827b4d502a7.pdf).

- It's common practice to randomly initialize parameters by drawing from independent zero-centered Gaussians.  One consideration is initializing $\phi$ to be sufficiently far from $0$ so our gradient doesn't explode (see in [Appendix](#verify-gradient-and-hessian-computations) that gradient magnitude is inversely proportional to $\phi$).  

- Personally, I like using numpy's `np.random.normal()` rather than TensorFlow's `tf.random_normal()` because I can check the generated value without using `sess.run()`.  But the TensorFlow-provided function can be used with a GPU, so for those interested:

```python
INIT_MU_PARAMS = {'mean': 0.0, 'stddev': 0.1}
INIT_PHI_PARAMS = {'mean': 1.0, 'stddev': 0.1}

mu = tf.Variable(initial_value=tf.random_normal(shape=[],
                                                seed=RANDOM_SEED,
                                                **INIT_MU_PARAMS),
                 dtype=tf.float32)
phi = tf.Variable(initial_value=tf.random_normal(shape=[],
                                                 seed=RANDOM_SEED,
                                                 **INIT_PHI_PARAMS),
                  dtype=tf.float32)
```

<a id="part2"></a>

#### Part 2:  Run optimization scheme

```python
LEARNING_RATE = 0.001
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=neg_log_likelihood)

with tf.Session() as sess:
	# initialize
    sess.run(fetches=tf.global_variables_initializer())

    i = 1
    obs_mu, obs_phi, obs_sigma = sess.run(fetches=[[mu], [phi], [sigma]])
    obs_loss = sess.run(fetches=[neg_log_likelihood], feed_dict={x: x_obs})
    obs_grad = sess.run(fetches=[grad], feed_dict={x: x_obs})
        
    while True:
        # gradient step
        sess.run(fetches=train_op, feed_dict={x: x_obs})

        # update parameters
        new_mu, new_phi, new_sigma = sess.run(fetches=[mu, phi, sigma])
        diff_norm = np.linalg.norm(np.subtract([new_mu, new_phi],
                                               [obs_mu[-1], obs_phi[-1]]))
        # update loss
        new_loss = sess.run(fetches=neg_log_likelihood, feed_dict={x: x_obs})
        loss_diff = np.abs(new_loss - obs_loss[-1])

        # update gradient
        new_grad = sess.run(fetches=grad, feed_dict={x: x_obs})
        grad_norm = np.linalg.norm(new_grad)
        
        obs_mu.append(new_mu)
        obs_phi.append(new_phi)
        obs_sigma.append(new_sigma)
        obs_loss.append(new_loss)
        obs_grad.append(new_grad)

        if param_diff_norm < TOL_PARAM:
            print('Parameter convergence in {} iterations!'.format(i))
            break

        if loss_diff < TOL_LOSS:
            print('Loss function convergence in {} iterations!'.format(i))
            break

        if grad_norm < TOL_GRAD:
            print('Gradient convergence in {} iterations!'.format(i))
            break

        if i >= MAX_ITER:
            print('Max number of iterations reached without convergence.')
            break

        i += 1
```

More notes:

- We used the Adam optimizer which is a pretty good default choice since it has a momentum-based adaptive step size.  I've noticed that traversing steep or flat regions of parameter space can be problematic since the Adam update rule depends on an averaging of past computed gradients.  You can see this effect by changing the $\phi$ initialization closer to $0$ which causes the gradient to explode.

	- See [Sebastian Ruder's blog post](http://sebastianruder.com/optimizing-gradient-descent/) for an overview of other (stochastic) gradient descent methods, though I think time is better spent worrying about improving the initialization scheme.

- The initial learning rate at $\alpha = 0.001$ and convergence tolerance values at $\delta = 10^{-8}$ are common default choices.  Standardizing the variables can make it easier to set reasonable defaults that work across multiple realizations of the generating distribution, but picking good values for $\alpha$ and $\delta$ is often a trial-and-error effort.

- When checking for parameter convergence, the norm of the parameter vector is computed using the values of $\phi$ even though we choose to display values of $\sigma$ for interpretability.

#### Results

```
  iter |     mu     |   sigma    |    loss    |    grad   
     1 | 0.17740524 | 1.07955348 | 107.118011 | 166.314087
   101 | 0.27897912 | 0.87825149 | 86.2192841 | 185.379456
   201 | 0.38198292 | 0.68829656 | 61.8442612 | 208.598816
   301 | 0.47547418 | 0.51341331 | 34.3445473 | 229.603500
   401 | 0.53297198 | 0.36388707 | 7.32367325 | 222.097382
   501 | 0.54181617 | 0.25903416 | -10.640480 | 137.181747
   601 | 0.54176593 | 0.21581791 | -14.554010 | 26.8282032
   701 | 0.54176593 | 0.20935123 | -14.655022 | 1.52713740

Loss function convergence in 718 iterations!

Fitted MLE: [0.5418, 0.2092]
Target MLE: [0.5418, 0.2090]
```

The optimization procedure indeed converges (within some floating point precision) to the sample mean and sample standard deviation of the transformed data --- it works!  The full code for this example is provided [here](https://github.com/kyleclo/tensorflow-mle/blob/master/univariate_gauss_adam.py).

![canon-adam-001]({{ site.url }}/images/posts/canon-adam-001.png)

## MLE of univariate Gaussian with Newton methods

What about Newton methods?  Even if most optimization problems are non-convex, I seems like a good idea to know how to call BFGS in TensorFlow.

#### Computing the Hessian

TensorFlow provides a `tf.hessians()` function that appears similar to `tf.gradients()` in its API, but I've found it extremely difficult to work with --- grumble grumble can only compute second derivatives with respect to one-dimensional Tensors.

Instead, I recommend something like this:

```python
hess = tf.stack(values=[tf.gradients(grad[0], [mu, phi]),
                        tf.gradients(grad[1], [mu, phi])], axis=0)
```

where `grad[0]` and `grad[1]` are the derivatives of `neg_log_likelihood` with respect to `mu` and `phi` respectively.  

For example, the final Hessian upon convergence is:

```
| 2285.12793 | -0.0032801 |
| -0.0032043 | 3814.80054 |
```

which we can verify is positive definite by checking:

- `np.linalg.cholesky(obs_hess[-1])` successfully returns a result, or
- `np.linalg.eigvals(obs_hess[-1])` returns all positive eigenvalues

where `obs_hess` is a list storing the computed Hessians at each iteration.   

#### Convexity of natural parameterization

Note that the computed Hessians are not always positive definite.  For example, `obs_hess[0]` is:

```
| 85.8048019 | 120.359650 |
| 120.359650 | -45.029373 |
```

While we can [prove](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf) that the natural parameter space for exponential families is always convex, this is not necessarily true for other parameterizations.

The distributions provided by `tf.contrib.distributions` don't have natural parameterization options, so they'll need to be coded from scratch.  Let's do this for the Gaussian distribution:

$$f_{\eta}(x) = h(x) \exp\left( \eta \cdot T(x) - A(\eta) \right)$$

where:

- The natural parameters $\eta =\left[\mu / 2 \sigma^2, -1 / 2 \sigma^2 \right]^T$
- The sufficient statistics $T(x) = \left[ x, x^2 \right]^T$
- The log-partition function $A(\eta) = -\eta_1^2/ 4 \eta_2 - \log (-2\eta_2) / 2$

Dropping the constant term $h(x)$, we replace our loss function with:

```python
log_partition = -0.25 * tf.square(eta1) / eta2 - 0.5 * tf.log(-2.0 * eta2)
log_prob = x * eta1 + tf.square(x) * eta2 - log_partition
neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)
```

where `eta1` and `eta2` are our defined parameters instead of `mu` and `phi`.

#### Using scipy's BFGS optimizer

Now that we have a convex loss function, we can use a Newton method for faster convergence.  TensorFlow provides an interface for external optimizers.  For example, we can use scipy's BFGS optimizer:

```python
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=neg_log_likelihood,
                                                   method='L-BFGS-B')
...

while True:
	optimizer.minimize(session=sess, feed_dict={x: x_obs})
```

Note the API is slightly different.  We're passing in the session into the optimizer's `minimize()` method instead of calling `sess.run()`.   The full code for this example is provided [here](https://github.com/kyleclo/tensorflow-mle/blob/master/univariate_gauss_bfgs.py).

Running it, we see the algorithm converges very quickly to the solution:

```
  iter |     mu     |   sigma    |    loss    |    grad   
     1 | 0.54175494 | 0.20899648 | -106.54919 | 0.00128950

Parameter convergence in 3 iterations!

Fitted MLE: [0.5418, 0.2090]
Target MLE: [0.5418, 0.2090]
```

Don't be confused by the output.  The optimization is taking gradients and hessians with respect to the natural parameters.  I simply like reporting $\mu$ and $\sigma$ because they're more interpretable.

![bfgs-musigma]({{ site.url }}/images/posts/bfgs-musigma.png)

![bfgs-eta]({{ site.url }}/images/posts/bfgs-eta.png)


## Appendix

#### Verify gradient and Hessian computations

Let's quickly verify that our code for computing gradients and Hessians are correct.  We derive analytically:

$$\mathcal{L}(\mu, \phi) = n \log \phi^2 + \frac{\sum (x_i - \mu)^2}{2\phi^4}$$

$$\nabla \mathcal{L}(\mu, \phi) = \begin{pmatrix} - \frac{\sum(x_i - \mu)}{\phi^4} & \frac{2n}{\phi} - \frac{2\sum(x_i - \mu)^2}{\phi^5} \end{pmatrix}$$

$$\text{Hess } \mathcal{L}(\mu, \phi) = \begin{pmatrix}
\frac{n}{\phi^4} & \frac{4 \sum(x_i - \mu)}{\phi^5} \\ \frac{4\sum(x_i - \mu)}{\phi^5} & -\frac{2n}{\phi^2} + \frac{10 \sum (x_i - \mu)^2}{\phi^6}
\end{pmatrix}$$

Now we can evaluate:

```python
def analytic_grad(x, mu, phi):
    n = x.size
    g1 = -1 * (x - mu).sum() / phi ** 4
    g2 = 2 * n / phi - 2 * ((x - mu) ** 2).sum() / phi ** 5
    return np.array([g1, g2])

def analytic_hess(x, mu, phi):
    n = x.size
    h11 = n / phi ** 4
    h21 = 4 * (x - mu).sum() / phi ** 5
    h12 = h21
    h22 = -2 * n / phi ** 2 + 10 * ((x - mu) ** 2).sum() / phi ** 6
    return np.array([[h11, h12], [h21, h22]])

print(analytic_grad(x=x_obs, mu=obs_mu[-1], phi=obs_phi[-1]))
print(analytic_hess(x=x_obs, mu=obs_mu[-1], phi=obs_phi[-1]))
```

which match the values of `obs_grad[-1]` and `obs_hess[-1]` (up to some floating point precision).  

#### Gradient descent with natural parameters

We performed gradient descent on the canonical parameters and BFGS on the natural parameters.  But is there any benefit to performing gradient descent on the natural parameters?  

Maybe.  

To illustrate, let's run Adam on the natural parameters (see the commented-out lines in BFGS example code) with initial learning rate $\alpha = 0.1$:

![nat-adam-01-musigma]({{ site.url }}/images/posts/nat-adam-01-musigma.png)

![nat-adam-01-eta]({{ site.url }}/images/posts/nat-adam-01-eta.png)

And we compare this to running Adam on the canonical parameters also with $\alpha = 0.1$:

![canon-adam-01]({{ site.url }}/images/posts/canon-adam-01.png)

For this choice of $\alpha$, we notice:

- The natural parameterization converges slower than the canonical parameterization.

- The natural parameterization seems to result in a smoothly-decreasing loss function whereas the loss under the canonical parameterization appears unstable.

- The gradient path under natural parameterization is a straight shot towards the location of the optima whereas the gradient path under canonical parameterization traverses parameter space in a more roundabout way.

And these empirical observations seem to hold for different choices of $\alpha$, though I still don't really have a great explanation for why.

#### Resources

[Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013, February). On the importance of initialization and momentum in deep learning. In International conference on machine learning (pp. 1139-1147).](http://proceedings.mlr.press/v28/sutskever13.pdf)

[Pinheiro, J. C., & Bates, D. M. (1996). Unconstrained parametrizations for variance-covariance matrices. Statistics and Computing, 6(3), 289-296.](https://pdfs.semanticscholar.org/2ff5/5b99d6d94d331670719bb1df1827b4d502a7.pdf)

[Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.](https://arxiv.org/pdf/1609.04747.pdf)
