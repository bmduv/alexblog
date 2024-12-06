---
title: Diffusion
draft: false
tags:
---
 
## Forward Diffusion:
- In a Markov Chain process of T steps (where each step depends on n previous step), add Gaussian Noise to current time step to generate next time step as shown by equation
$$
\begin{gathered}
q(x_t|x_t-1) = \mathcal{N}(x_t;mean_t = \sqrt{1-\beta_tx_t}, \Sigma_t = \beta_tI) \\\\
\text{Sample } \epsilon \sim \mathcal{N}(0, I) \text{ and then set } x_t = \sqrt{1 - \beta_tx_{t-1}} + \sqrt{\beta_t\epsilon}
\end{gathered}
$$
      Note: $\beta_t$ at each time step is not constant, variance scheduler is utilized (typically linear, quadratic, cosine)
- Reparameterization Trick:
$$
\begin{gathered}
a_t = 1 - \beta_t \\
\text{Scaling factor that depends on noise variance} \\\\
\bar{a}_t = \prod_{s=0}^t\alpha_s \\
\text{Prodcut of all alpha values from step 0 to step t,} \\
\text{represents cumulative effect of scaling up to timestep t} \\\\
\epsilon{0}, \epsilon{1}, ..., \epsilon{t} - 1 ~N(0, I)\\
\text{Independent Gaussian noise variables drawn from normal} \\
\text{distribution with mean 0 and identify covariance matrix I} \\\\
x_t = \bar{a}_t x_0 + \sqrt{1 - \bar{a}_t \epsilon_0} \\
\text{Allows us the express } x_t \text{ in terms of initial data point} \\
x_0 \text{ and a series of Gaussian noise terms}
\end{gathered}
$$
   Note: Since $\beta_t$ is a hyperparameter, we can precompute scaling factor and the cumulative scaling factor for timestep t for all timesteps
## Reverse Diffusion:
- Since $x_t$ is nearly an isotropic Gaussian distribution as $T \rightarrow \infty$ we can start by sampling $x_t$ from Normal distribution $\mathcal{N}(0, I)$ 
- Using reverse distribution $p(x_t-1 | x_t)$, iteratively sample $x_t-1$ from $x_t$ moving backward from $T$ to 0
   - $p(x_t-1 | x_t)$ is intractable to know since it requires knowing the distribution of all possible images to calculate conditional probability $\rightarrow$ we leverage neural network to learn the conditional probability distribution called $p_\theta(x_{t-1} | x_t)$ with $\theta$ representing the parameters of the neural network
$$
\begin{gathered}
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) \\
\text{Neural network needs to learn the mean and variance}
\end{gathered}
$$
- Objective Function:
   - Similar to VAE, we can use variational lower bound (ELBO) to minimize negative log-likelihood with respect to ground truth data sample $x_0$ 