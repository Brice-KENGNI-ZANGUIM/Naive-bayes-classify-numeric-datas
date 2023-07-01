############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
from dataclasses import dataclass

############################################################################################################
##########################         CLASSES TO HOLD DITRIBUTIONS PARAMETERS        ##########################
############################################################################################################

@dataclass
class params_gaussian:
    mu: float
    sigma: float
        
    def __repr__(self) :
        return f"params_gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"

@dataclass
class params_binomial:
    n: int
    p: float
        
    def __repr__(self) -> str:
        return f"params_binomial(n={self.n:.3f}, p={self.p:.3f})"

@dataclass
class params_uniform:
    a: int
    b: int
        
    def __repr__(self) -> str:
        return f"params_uniform(a={self.a:.3f}, b={self.b:.3f})"
    
@dataclass
class params_poisson: 
    lamda : float

    def __repr__(self) -> str:
        return f"params_poisson(lamda={self.lamda:.3f})"