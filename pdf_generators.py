############################################################################################################
##########################                   MODULES IMPORTATION                  ##########################
############################################################################################################

import numpy as np
from math import factorial, gamma
import scipy

############################################################################################################
##########################                   FUNCTIONS DEFINITION                 ##########################
############################################################################################################

def pdf_poisson( x, lamda ): 

    return lamda**x/gamma(x+1)*np.exp(-lamda)

def pdf_uniform(x, a, b):
    """
    Calculates the probability density function (PDF) for a uniform distribution between 'a' and 'b' at a given point 'x'.

    Args:
        x (float): The value at which the PDF is evaluated.
        a (float): The lower bound of the uniform distribution.
        b (float): The upper bound of the uniform distribution.

    Returns:
        float: The PDF value at the given point 'x'. Returns 0 if 'x' is outside the range [a, b].
    """
    ### START CODE HERE ###
    pdf = 1/(b-a) if a <= x <= b else 0 
    ### END CODE HERE ###

    return pdf

def pdf_gaussian(x, mu, sigma):
    """
    Calculate the probability density function (PDF) of a Gaussian distribution at a given value.

    Args:
        x (float or array-like): The value(s) at which to evaluate the PDF.
        mu (float): The mean of the Gaussian distribution.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float or ndarray: The PDF value(s) at the given point(s) x.
    """

    ### START CODE HERE ###
    pdf = np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    
    ### END CODE HERE ###

    return pdf

def pdf_binomial(x, n, p):
    """
    Calculate the probability mass function (PMF) of a binomial distribution at a specific value.

    Args:
        x (int): The value at which to evaluate the PMF.
        n (int): The number of trials in the binomial distribution.
        p (float): The probability of success for each trial.

    Returns:
        float: The probability mass function (PMF) of the binomial distribution at the specified value.
    """

    ### START CODE HERE ###
    pdf = scipy.special.comb(n,x)*p**x*(1-p)**(n-x)
    ### END CODE HERE ###

    return pdf


