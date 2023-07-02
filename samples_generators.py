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
    DESCRIPTION :
    ------------
        Generates an array of uniformly distributed random numbers within the specified range [a,b[

    PARAMETERS :
    -----------
        - a : float
             The lower bound of the range.
        - b : float 
            The upper bound of the range.
        - num_samples : int 
            The number of samples to generate (default: 100).

    RETURNS  :
    ---------
        - array : ndarray
             An array of random numbers sampled uniformly from the range [a, b).
    """

    array = np.random.uniform(a , b , num_samples)

    return array

def inverse_cdf_gaussian(y, mu, sigma):
    """
    DESCRIPTION :
    ------------
        Calculates the inverse cumulative distribution function (CDF) of a Gaussian distribution.

    PARAMETERS :
    -----------
        - y : float or ndarray
             The probability or array of probabilities.
        - mu : float   
            The mean of the Gaussian distribution.
        - sigma : float
            The standard deviation of the Gaussian distribution.

    RETURNS :
        - x : float or ndarray
             The corresponding value(s) from the Gaussian distribution that correspond to the given probability/ies.
    """

    x = mu + sigma*np.sqrt(2)*scipy.special.erfinv(2*y-1)

    return x

def gaussian_generator(mu, sigma, num_samples):
    """
    DESCRIPTION :
    ------------
        Generate Gaussian distribute sample with specific parameter.

    PARAMETERS :
    -----------
        - mu : float   
            The mean of the Gaussian distribution.
        - sigma : float
            The standard deviation of the Gaussian distribution.
        - num_sample : int
            size of the sample data to generate 

    RETURNS :
        - x : ndarray
            The corresponding array containing gaussian distributes datas
    """
    # Array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0 , 1 , num_samples)

    # with uniform-distributed sample, generate Gaussian-distributed data from the inverse of Gaussian CDF ( Cumulated Distribution function)
    array = inverse_cdf_gaussian(u , mu, sigma)

    return array

def inverse_cdf_binomial(y, n, p):
    """
    DESCRIPTION : 
    ------------
        Calculates the inverse cumulative distribution function (CDF) of a binomial distribution.

    PARAMETERS :
    -----------
        - y : float or ndarray
             The probability or array of probabilities.
        - n :int
             The number of trials in the binomial distribution.
        - p : float
             The probability of success in each trial.

    RETURNS :
    --------
        - x : float or ndarray
             The corresponding value(s) from the binomial distribution that correspond to the given probability/ies.
    """
    if not isinstance(n , int ) : n = int(n) +1
    x = binom.ppf(y, n, p)

    return x

def binomial_generator(n, p, num_samples):
    """
    DESCRIPTION :
    ------------
        Generates an array of binomially distributed random numbers.

    PARAMETERS :
    -----------
        n : int
            The number of trials in the binomial distribution.
        p : float
            The probability of success in each trial.
        num_samples : int
             The number of samples to generate.

    RETURNS :
    -------
        array:  
            An array of binomially distributed random numbers.
    """
    if not isinstance(n , int ) : n = int(n) +1
    # First generate an array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0 , 1 , num_samples)

    # Use the uniform-distributed sample to generate binomial-distributed data
    array = inverse_cdf_binomial(u , n, p)
    
    return array

def poisson_generator ( lamda, num_samples): 
    """
    DESCRIPTION :
    ------------
        Generates an array of poisson  distributed random numbers.

    PARAMETERS :
    -----------
        lamda : float
            Corresponding mean value of the poisson distribution
        num_samples : int
             The number of samples to generate.

    RETURNS :
    -------
        array:  
            An array of poisson distributed random numbers.
    """
    return np.random.poisson(lamda, num_samples)