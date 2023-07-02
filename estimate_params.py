############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
import numpy as np
import math

############################################################################################################
##########################                   FUNCTIONS DEFINITION                 ##########################
############################################################################################################

def estimate_gaussian_params(sample):
    """
    DESCRIPTION :
    ------------
        The function estimate the gausian parameter of a sample; assuming the datas in sample are gaussian distributed
        The parameters to estime are the mean 'mu' and the standard deviation 'sigma'

    PARAMETERS :
    -----------
        - sample : list or nd-array
            datas sample we will like to fit/estimate as gaussian distributed

    RETURNS :
    --------
        - mu : float
            estimate mean value of the gaussian distribution 
        - sigma : float
            estimate standard deviation of the gaussian distribution
    """

    mu = np.mean(sample)

    sigma = np.std(sample)

    return mu, sigma

def estimate_binomial_params(sample):
    """
    DESCRIPTION :
    ------------
        The function estimate the binomial parameter of a sample; assuming the datas in sample are binomial distributed
        The parameters to estime are the number of trials 'n' and the probability of success deviation 'p'

    PARAMETERS :
    -----------
        - sample : list or nd-array
            datas sample we will like to fit/estimate as binomial distributed

    RETURNS :
    --------
        - n : int
            estimate number of trial of the binomial distribution
        - p : float
            estimate probability of success
    """

    mu = np.mean(sample)
    variance = np.var(sample)

    p = 1 - variance/mu
    if p == 0 or math.isnan(mu) or math.isnan(p) :
        n = 10
    else :
        n = int(mu/p) + 1 

    return n, p

def estimate_uniform_params(sample):
    """
    DESCRIPTION :
    ------------
        The function estimate the uniform parameter of a sample; assuming the datas in sample are uniformly distributed
        The parameters to estime are the lower value 'a' and the upper value deviation 'b'

    PARAMETERS :
    -----------
        - sample : list or nd-array
            datas sample we will like to fit/estimate as uniform distributed

    RETURNS :
    --------
        - a : float
            estimate lower value of the sample 
        - b : float
            estimate upper value of the sample
    """
    
    a = sample.min()
    b = sample.max()

    return a, b

def estimate_poisson_params ( sample ) :
    """
    DESCRIPTION :
    ------------
        The function estimate the poisson parameter of a sample; assuming the datas in sample are poisson distributed
        The parameters to estime are the mean value 'lamda'

    PARAMETERS :
    -----------
        - sample : list or nd-array
            datas sample we will like to fit/estimate as poisson distributed

    RETURNS :
    --------
        - lamda : float
            estimate mean value of the sample ; the only parameter of a poisson distribution

    """
    
    lamda = np.mean(sample)

    return lamda

