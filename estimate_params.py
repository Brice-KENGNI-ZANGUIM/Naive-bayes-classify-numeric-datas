import numpy as np

def estimate_gaussian_params(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)

    return mu, sigma

def estimate_binomial_params(sample):
    mu = np.mean(sample)
    variance = np.var(sample)

    p = 1 - variance/mu
    n = int(mu/p) + 1 

    return n, p

def estimate_uniform_params(sample):
    a = sample.min()
    b = sample.max()

    return a, b

def estimate_poisson_params ( sample ) :
    lamda = np.mean(sample)

    return lamda

