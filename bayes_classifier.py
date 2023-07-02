############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
import numpy as np
from configurations import classes_params
from datas_generators import get_distribution
from estimate_params import (  estimate_binomial_params, estimate_gaussian_params, estimate_poisson_params, estimate_uniform_params )
from params_classes import ( params_binomial, params_gaussian, params_poisson, params_uniform )
from pdf_generators import ( pdf_binomial, pdf_gaussian, pdf_poisson, pdf_uniform )

from configurations import FEATURES, CLASSES

############################################################################################################
##########################                     DEFINE FUNCTIONS                   ##########################
############################################################################################################

def compute_training_params(data, features= FEATURES, classes = CLASSES):
    """
    PARAMETERS :
    -----------
        Computes the estimated parameters for training a model based on the provided dataframe and features.

    PARAMETERS :
    -----------
        - data : pandas.DataFrame
            The dataframe containing the training data.
        - features : list)
            A list of feature names to consider for the probability computation.
        - classes : list
            A list of classes present in our data provide by 'data'

    RETURNS :
    --------
        tuple : A tuple containing two dictionaries:
            - params_dict : dict
                A dictionary that contains one part of the estimate posterior probability for each class and feature : P(feature|class)
            - probs_dict : dict
                A dictionary that contains the proportion of data belonging to each class.
    """
    
    # Dict that should contain the estimated parameters
    params_dict = {}
    
    # Dict that should contain the proportion of data belonging to each class
    probs_dict = {}
        
    # Loop over the breeds
    for classe in classes :
        
        # Slice the original df to only include data for the current class and the feature columns
        sub_data = data[data["class"] == classe][features]
        
        # Save the probability of each class (breed) in the probabilities dict
        probs_dict[classe] = sub_data.shape[0]/data.shape[0]
        
        # Initialize the inner dict
        inner_dict = {} 
        
        # Loop over the columns of the sliced dataframe
        for feature in features:
            distrib = get_distribution(classes_params[classe][feature].__repr__( ) )
            match distrib:
                case "gaussian": 
                    # Estimate parameters assuming a gaussian distribution of the current feature
                    mu , sigma = estimate_gaussian_params( sub_data[feature] )
                    estimate_param = params_gaussian( mu = mu , sigma = sigma )
                    
                case "binomial":
                    # Estimate parameters assuming a binomial distribution of the current feature
                    n , p = estimate_binomial_params( sub_data[feature] )
                    estimate_param = params_binomial( n = n , p = p)
                    
                case "uniform":
                    # Estimate parameters assuming a uniform distribution of the current feature
                    a , b = estimate_uniform_params( sub_data[feature] )
                    estimate_param = params_uniform( a = a , b = b)

                case "poisson":
                    # Estimate parameters assuming a uniform distribution of the current feature
                    lamda = estimate_poisson_params( sub_data[feature] )
                    estimate_param = params_poisson( lamda = lamda)

            # Save the dataclass object within the inner dict
            inner_dict[feature] = estimate_param
        
        # Save inner dict within outer dict
        params_dict[classe] = inner_dict
    
    return params_dict, probs_dict


def prob_of_X_given_class( X, classe, params_dict, features = FEATURES ):
    """
    DESCRIPTION :
    ------------
        Calculate the conditional probability of X given a specific class, using the given features and parameters.

    PARAMETERS :
    -----------
        - X : list
            List of feature values of an observable for which the probability needs to be calculated.
        - classe : str
            The class for which the probability is calculated.
        - params_dict : dict
            Dictionary containing the parameters for different breeds and features.
        - features : list
            List of feature names corresponding to the feature values in X.

    RETURNS :
    --------
        float: The conditional probability of X given the specified class 'classe'.
    """
    
    # To assure that the list of feature 'features' name and the provides values X have the same size 
    if len(X) != len(features):
        print("X and list of features should have the same length")
        return 0

    # We initialize the probability  to 1
    probability = 1.
        
    for x_feat, feature in zip( X, features ):
        
        # Get the relevant parameters from params_dict 
        params = params_dict[classe][feature]
        distrib = get_distribution( params.__repr__( ) )
        match distrib:
            # You can add add as many case statements as you see fit
            case "gaussian": 
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_gaussian(x_feat, params.mu , params.sigma)
                
            case "binomial": 
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_binomial( x_feat, params.n, params.p )

            case "uniform": 
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_uniform( x_feat, params.a, params.b )

            case "poisson":
                probability_f = pdf_poisson( x_feat, params.lamda )
        
        # Multiply by probability of current feature
        probability *= probability_f
            
    return probability

def predict_class(X, params_dict, probs_dict):
    """
    DESCRIPTION :
    ----------
        Predicts the breed based on the input and features.

    PARAMETERS :
    -----------
        - X : array-like
            The input data for prediction.
        - params_dict : dict
            A dictionary containing parameters for different classes.
        - probs_dict : dict
            A dictionary containing probabilities for different classes.

    RETURNS :
    --------
        str : The predicted class name.
    """

    # Calcul all the probabilities for every classes
    predict_class_probs = [ prob_of_X_given_class(X, classe, params_dict)*probs_dict[classe] for classe in CLASSES]
    
    # get the class with the maximum posterior
    prediction = np.argmax(predict_class_probs)
    
    return CLASSES[prediction]