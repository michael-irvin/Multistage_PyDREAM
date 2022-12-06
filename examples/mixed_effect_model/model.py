import pandas as pd
import numpy as np
from pydream.parameters import SampledParam
from scipy.stats import norm, invgauss, uniform

# Simple random effects model, example.

# Data
data = pd.read_csv("/Users/michaelirvin/Research/Multistage_PyDREAM/examples/mixed_effect_model/data.csv")

# Model
#Formula 1: $$ p(\mu, \tau, \pmb\theta, | \mathscr D, \sigma) = p(\mu, \tau) \Pi_{j=1}^{N}p(x_{i,j}|\theta_j, \sigma) \Pi_{j=1}^{M}p(\theta_j|\mu, \tau) $$
# p1 = \Pi_{j=1}^{M}p(\theta_j|\mu, \tau); antibody effect: normal distribution centered at \mu and scaled by \tau
# p2 = \Pi_{j=1}^{N}p(x_{i,j}|\theta_j, \sigma); patient effect: log-normal distribution centered 10^{\theta_j} and scaled by {\sigma}

# priors
shrinkage_term = 0.2 # \sigma
mu_prior = data['expression'].mean()
tau_prior = data['expression'].std() - shrinkage_term

sampled_priors = [ # mu and tau
    SampledParam(norm, loc=mu_prior, scale=1.5),
    SampledParam(invgauss, *[tau_prior], scale=2.1)
] + [ # antibody effect (\theta)
    SampledParam(uniform, loc=-10, scale=20),  # Uniform pdf bounded [-10, 10]
    SampledParam(uniform, loc=-10, scale=20),
    SampledParam(uniform, loc=-10, scale=20),
    SampledParam(uniform, loc=-10, scale=20),
    SampledParam(uniform, loc=-10, scale=20),
] 
# log ps
def p1(x): #ps antibody term
    mu = x[0]
    tau = x[1]
    p_theta_j = norm(loc=mu, scale=tau).pdf(x[2:])  
    return np.sum(np.log(p_theta_j))


antibodies = sorted(data['antibody'].unique()) 
def p2(x):
    log_p_theta = p1(x)
    # patients value probability
    # p(log(x_{i,j})) = N(theta_j, sigma)
    log_ps = 0
    for i, antibody in enumerate(antibodies):
        idx = i + 2
        log_expression_levels = np.log(data[data.antibody==antibody].expression.values)
        p_log_expression_levels = norm(loc=x[idx], scale=shrinkage_term).pdf(log_expression_levels)
        log_ps += np.sum(np.log(p_log_expression_levels))
    return log_ps + log_p_theta