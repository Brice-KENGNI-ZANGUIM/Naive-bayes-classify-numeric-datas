#  DESCRIPTION :
#  ------------
#      The configuration of your build-in datas
#      Contains all datas informations and variables needed to build the datas to train de Naïve bayes model
#
#  AUTHOR : 
#  -------
#      Name : Brice KENGNI ZANGUIM
#      mail : kenzabri2@yahoo.com
#
############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
from params_classes import  (
    params_binomial, params_gaussian, params_poisson, params_uniform 
    )
import numpy as np
############################################################################################################
##########################                   VARIABLES DEFINITION                 ##########################
############################################################################################################

# Define the seed parameter for 
np.random.seed(1024)

# List of every features to create : it could be lenght , heigh, age, planet radius, mass, salary, 
FEATURES = [ "fetaure_gauss_1", "fetaure_gauss_2", "fetaure_binom_3", "fetaure_poisson_4", "fetaure_uniform_5" ]

#liste of differents classes to classify datas: could be a list of dog breeds, horse breeds, plant breeds, stars type, SPAM-HAM . . .etc 
# you can add more classes as you need or remove some; but by doing that make sure you also update the dictionary 'classes_params' below
CLASSES = [ "class_1", "class_2", "class_3", "class_4", "class_5" ]

# number of sample to generate for every class
N_SAMPLE = [ 50_0, 50_0, 50_0 , 50_0, 50_0 ]

# check if variables CLASSES and N_SAMPLE have the same size
if len(N_SAMPLE) != len(CLASSES) :
    raise ValueError ( f"the lengh of variable 'N_SAMPLE' should be egal that of variable 'CLASSES' : {len(N_SAMPLE)} != {len(CLASSES)} " )

############################################################################################################
##########################                      CLASSES PARAMS                    ##########################
############################################################################################################
# parameters here define related too the probability distribution function we will like every class's feature to look like
# You can add more classes or remove some but if you do so make sure  to update the above 'CLASSES' variable

mult = 1.00001
classes_params = {
                CLASSES[0]: {
                    FEATURES[0] : params_gaussian( mu=44, sigma=2.42 ),
                    FEATURES[1] : params_gaussian( mu=37, sigma=0.85 ),
                    FEATURES[2] : params_binomial( n=15, p=0.2 ),
                    FEATURES[3] : params_poisson( lamda= 3.4 ),
                    FEATURES[4] : params_uniform( a=0.1, b=0.8 )
                            },
                
                CLASSES[1]: {
                    FEATURES[0] : params_gaussian( mu=15, sigma=1.8 ),
                    FEATURES[1] : params_gaussian (mu=39, sigma=3.2 ),
                    FEATURES[2] : params_binomial( n=30, p=0.6 ),
                    FEATURES[3] : params_poisson( lamda = 4.7 ),
                    FEATURES[4] : params_uniform( a=0.5, b=1.3 )
                            },
                        
                CLASSES[2]: {
                    FEATURES[0] : params_gaussian(mu=45, sigma=1.2),
                    FEATURES[1] : params_gaussian(mu=25, sigma=2.3),
                    FEATURES[2] : params_binomial(n=50, p=0.8),
                    FEATURES[3] : params_poisson(lamda= 1.2),
                    FEATURES[4] : params_uniform(a=0.3, b=0.6)
                            },
                        
                CLASSES[3]: {
                    FEATURES[0] : params_gaussian(mu=28, sigma=1.42),
                    FEATURES[1] : params_gaussian(mu=11, sigma=0.7),
                    FEATURES[2] : params_binomial(n=10, p=0.45),
                    FEATURES[3] : params_poisson(lamda= 2.17),
                    FEATURES[4] : params_uniform(a=1.15, b=1.8)
                            },
                        
                CLASSES[4]: {
                    FEATURES[0] : params_gaussian(mu=28*mult, sigma=1.42*mult),
                    FEATURES[1] : params_gaussian(mu=11*mult, sigma=0.7*mult),
                    FEATURES[2] : params_binomial(n=10*mult, p=0.45),
                    FEATURES[3] : params_poisson(lamda= 2.17*mult),
                    FEATURES[4] : params_uniform(a=1.15*mult, b=1.8*mult)
                            }
            }

if __name__ == "__main__" :
    print(classes_params[CLASSES[0]][FEATURES[0]].__repr__()) 