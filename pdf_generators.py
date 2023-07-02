# This modul contains analytical form of many probability distribution functions ( PDF )

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
    """
    DESCRIPTION :
    ------------
        Calculates the probability density function (PDF) for a poisson distribution at a given point 'x'.

    PARAMETERS :
    -----------
        - x : float
            The value at which the PDF is evaluated.
        - lamda : float
            The lambda parameter of the poisson distribution function representing the mean value

    RETURNS :
    --------
        float: The PDF value at the given point 'x'
    """
    if not isinstance(x , int ) : x = int(x) +1

    pdf_x = lamda**x/gamma(x+1)*np.exp(-lamda)

    return pdf_x

def pdf_uniform(x, a, b):
    """
    DESCRIPTION :
    ------------
        Calculates the probability density function (PDF) for a uniform distribution between 'a' and 'b' at a given point 'x'.

    PARAMETERS :
    -----------
        - x : float
            The value at which the PDF is evaluated.
        - a : float
            The lower bound of the uniform distribution.
        - b : float
            The upper bound of the uniform distribution.

    RETURNS :
    --------
        float: The PDF value at the given point 'x'. Returns 0 if 'x' is outside the range [a, b].
    """
    pdf_x = 1/(b-a) if a <= x <= b else 0 

    return pdf_x

def pdf_gaussian(x, mu, sigma):
    """
    DESCRIPTION :
    ------------
        Calculate the probability density function (PDF) of a Gaussian distribution at a given value.

    PARAMETERS :
        - x : float or array-like
            The value(s) at which to evaluate the PDF.
        - mu : float
            The mean of the Gaussian distribution.
        - sigma : float
            The standard deviation of the Gaussian distribution.

    RETURNS :
    --------
        float or ndarray: The PDF value(s) of a gaussian function at the given point(s) x.
    """

    pdf_x = np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))
    

    return pdf_x

def pdf_binomial(x, n, p):
    """
    DESCRIPTION :
    ------------
        Calculate the probability mass function (PMF) of a binomial distribution at a specific value.

    PARAMETERS :
    -----------
        - x : int
            The value at which to evaluate the PMF.
        - n : int
            The number of trials in the binomial distribution.
        - p : float
            The probability of success for each trial.

    RETURNS :
        float: The probability mass function (PMF) of the binomial distribution at the specified value.
    """
    if not isinstance(n , int ) : n = int(n) +1

    pdf_x = scipy.special.comb(n,x)*p**x*(1-p)**(n-x)

    return pdf_x
