# Flow Matching
>
> MIT [Lecture Notes](<https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf>)

$u_{t}^{\theta}(x)$ is the parameterized vector field. 

$\mathbb{R}^{d}$ represents data.

$\psi_t$ represents the flow of a vector field $u_t(x)$

## Training

Main condition we are trying to optimize is the following:
$\mathcal{L}_{CFM} = \lVert u_{t}^{\theta}(x) - u_{t}^{\text{target}}(x | z) \lVert^{2}$

$\nabla_{\theta} \mathcal{L}_{FM} = \nabla_{\theta} \mathcal{L}_{CFM} $

When we use the **CondOT probability** path which is literally linear interpolation from noise to data, our loss becomes something even simpler:

$u_{t}^{\text{target}}(x | z) \lVert^{2} = (z - \epsilon)$

$\boxed{\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z\sim p_{\text{data}}, \epsilon\sim\mathcal{N}(0, I_d)} \left[\lVert u_{t}^{\theta}(tz + (1-t)\epsilon) - (z - \epsilon) \right] }$

In essence, $\epsilon = p_{\text{init}} = X_{0}$


### note:

I have no clue what the marginilization trick does or conditional marganilization has to do with this. Somehow, we get this general formula and using our probability path, we get an extremely simply L2 velocity training objective.

## Sampling

We use numerical methods as it not always possible to compute the "*flow $\psi_t$ explicity if $u_t$ is not as simple*". (page 8)

### ODE
For an ODE, we simply follow the vector field $u_t$ in small steps in the direction of each vector.

Steps (GIVEN $n$ STEPS):

1. Set $t=0$
2. Set step size $h = \frac1n$
3. Sample $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$

4. for loop $n$ steps:
5. $X_{t+h} = X_{t} + h u_{t}^{\theta}(X_t)$
6. $t = t + h$
7. end for
8. return $X_1 \sim p_{\text{data}}$

### SDE

$dX_t = u_t(X_t) dt + \sigma_t dW_t$

Only difference with SDE is the Brownian Wiener process and we simply also add that simulated process with the following term: $\sqrt{h} \epsilon_t$ We may also scale this diffusion term by $\sigma_t$. Combining this, we now have an algorithm for sampling an SDE.

Big issue with SDE is that their paths are no where differentiable.

Steps (GIVEN $n$ STEPS and diffusion coefficient $\sigma_t$):

1. Set $t=0$
2. Set step size $h = \frac1n$
3. Sample $X_0 \sim p_{\text{init}} = \mathcal{N}(0, I_d)$

4. for loop $n$ steps:
5. $\epsilon_t \sim \mathcal{N}(0, I_d)$
6. $X_{t+h} = X_{t} + h u_{t}^{\theta}(X_t) + \sigma_t\sqrt{h} \epsilon_t$
7. $t = t + h$
8. end for
9. return $X_1 \sim p_{\text{data}}$

## Skipped Score Matching

## Conditioning and CFG

$y \in \mathcal{Y}$, our conditionion variable or prompt $y$ lives in a space $\mathcal{Y}$.

When we are conditioning our model $u_{t}^{\theta}(x | y) : \mathbb{R}^{d} \times \mathcal{Y} \times [0, 1] \rightarrow \mathbb{R}^{d}, (x,y,t) \rightarrow u_{t}^{\theta}(x | y)$. Then we just add that our training objective which becomes the following: 

$\boxed{\mathcal{L}_{CFM}^{\text{guided}}(\theta) = \mathbb{E}_{t \sim \text{Unif}[0,1], (z,y)\sim p_{\text{data}}, x\sim p_t(\cdot | z)} \left[\lVert u_{t}^{\theta}(x | y) - u_{t}^{\text{target}}(x | z)\right]}$

 Sure it works but "*it was soon empirically realized that images samples with this procedure did not fit well enough to the desired label y (see Figure 11). This can have a diversity of reasons: the **model might underfit** (i.e. we do not actually learn the true marginal vector field) or our data might be imperfect (e.g. text-image pairs from the world wide web have a lot of errors). Therefore, to truly generate samples that fit better to a prompt, we have to find a way to artificially reinforce the prompt variable y.*" (page 35)

 So we use a different approached, namely Classifier-Free Guidance. Some black magic occurs that allows us to combine both features of null classification and our wanted classification. It is an heurstic.

 During training, we take $y$ and drop-out at a probability of $20\%$ which is empirically shown to perform the best. Then, during sampling, we use the folloiwng formula:

 $X_{1} = X_{0} + \int_{0}^{1} ((1-w) u_t(x|\varnothing) + w u_t(x|y)) \ dx$

 $\boxed{\tilde{u}_t(x|y) = (1-w) u_t(x|\varnothing) + w u_t(x|y)}.$