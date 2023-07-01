############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################

import numpy as np
import scipy
from scipy.stats import  binom

############################################################################################################
##########################                      DEFINE FUNCTIONS                  ##########################
############################################################################################################

def uniform_generator(a, b, num_samples=100):
    """
    Generates an array of uniformly distributed random numbers within the specified range.

    Parameters:
    - a (float): The lower bound of the range.
    - b (float): The upper bound of the range.
    - num_samples (int): The number of samples to generate (default: 100).

    Returns:
    - array (ndarray): An array of random numbers sampled uniformly from the range [a, b).
    """

    array = np.random.uniform(a , b , num_samples)

    return array

def inverse_cdf_gaussian(y, mu, sigma):
    """
    Calculates the inverse cumulative distribution function (CDF) of a Gaussian distribution.

    Parameters:
    - y (float or ndarray): The probability or array of probabilities.
    - mu (float): The mean of the Gaussian distribution.
    - sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    - x (float or ndarray): The corresponding value(s) from the Gaussian distribution that correspond to the given probability/ies.
    """

    x = mu + sigma*np.sqrt(2)*scipy.special.erfinv(2*y-1)

    return x

def gaussian_generator(mu, sigma, num_samples):

    # Array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0 , 1 , num_samples)

    # with uniform-distributed sample, generate Gaussian-distributed data from the inverse of Gaussian CDF ( Cumulated Distribution function)
    array = inverse_cdf_gaussian(u , mu, sigma)

    return array

def inverse_cdf_binomial(y, n, p):
    """
    Calculates the inverse cumulative distribution function (CDF) of a binomial distribution.

    Parameters:
    - y (float or ndarray): The probability or array of probabilities.
    - n (int): The number of trials in the binomial distribution.
    - p (float): The probability of success in each trial.

    Returns:
    - x (float or ndarray): The corresponding value(s) from the binomial distribution that correspond to the given probability/ies.
    """

    x = binom.ppf(y, n, p)

    return x

def binomial_generator(n, p, num_samples):
    """
    Generates an array of binomially distributed random numbers.

    Args:
        n (int): The number of trials in the binomial distribution.
        p (float): The probability of success in each trial.
        num_samples (int): The number of samples to generate.

    Returns:
        array: An array of binomially distributed random numbers.
    """

    # Generate an array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0 , 1 , num_samples)

    # Use the uniform-distributed sample to generate binomial-distributed data
    # Hint: You need to sample from the inverse of the CDF of the distribution you are generating
    array = inverse_cdf_binomial(u , n, p)

    return array

def poisson_generator ( lamda, num_samples): 

    return np.random.poisson(lamda, num_samples)