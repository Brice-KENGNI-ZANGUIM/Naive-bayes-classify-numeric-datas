############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
from dataclasses import dataclass

############################################################################################################
##########################         CLASSES TO HOLD DITRIBUTIONS PARAMETERS        ##########################
############################################################################################################

@dataclass
class params_gaussian:
    """
    DESCRIPTION :
    ------------
        A class to hold the gaussian distribution parameters

    PARAMETERS :
    -----------
        - mu : float
            mean value of the distribution function
        - sigma : float
            standard deviation of the distribution function
    """
    mu: float
    sigma: float
        
    def __repr__(self) :
        return f"params_gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"

@dataclass
class params_binomial:
    """
    DESCRIPTION :
    ------------
        A class to hold the binomial distribution parameters

    PARAMETERS :
    -----------
        - n : int
            The number of trial for the binomial distribution function
        - p : float
            the probability of success of the binomial distribution function
    """
    n: int
    p: float
        
    def __repr__(self) -> str:
        return f"params_binomial(n={self.n:.3f}, p={self.p:.3f})"

@dataclass
class params_uniform:
    """
    DESCRIPTION :
    ------------
        A class to hold the uniform distribution parameters

    PARAMETERS :
    -----------
        - a : float
            lower value of the uniform distribution function
        - b : float
            upper value of the uniform distribution function
    """
    a: int
    b: int
    
    def __repr__(self) -> str:
        return f"params_uniform(a={self.a:.3f}, b={self.b:.3f})"
    
@dataclass
class params_poisson:
    """
    DESCRIPTION :
    ------------
        A class to hold the poisson distribution parameters

    PARAMETERS :
    -----------
        - lamda : float
            mean value of the poisson distribution function
    """
    lamda : float

    def __repr__(self) -> str:
        return f"params_poisson(lamda={self.lamda:.3f})"