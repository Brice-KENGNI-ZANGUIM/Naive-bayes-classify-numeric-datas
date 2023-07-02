############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################
import pandas as pd
from samples_generators import (
                                gaussian_generator,
                                uniform_generator,
                                poisson_generator,
                                binomial_generator
                            )

from configurations import (
                            FEATURES, 
                            CLASSES, 
                            N_SAMPLE, 
                            classes_params
                        )

############################################################################################################
##########################                   FUNCTIONS DEFINITION                 ##########################
############################################################################################################

def get_distribution ( repr ):
    """
    DESCRIPTION :
    -----------
        provide the name of the probability distribution by analysing the '__repr__()' output

    PARAMETERS :
    -----------
        - repr : str  
            the output of the method '__repr()' from the 'dataclass' params

    RETURN :
    ------
        - str : 
            The name of the distribution. The output can take the value 'gaussian', 'poisson', 'binomial' or 'uniform'

    """
    if "gaussian" in repr and "mu=" in repr and "sigma=" in repr :
        distrib =  "gaussian"
    elif "uniform" in repr and "a=" in repr and "b=" in repr :
        distrib =  "uniform"
    elif "binomial" in repr and "n=" in repr and "p=" in repr :
        distrib = "binomial"
    elif "poisson" in repr and "lamda=" in repr :
        distrib = "poisson"

    return distrib


def split_datas ( dataframe , frac = (.7 , .3 ) ) :
    """
    DESCRIPTION :
    -----------
        Take a DateFrame and split into to differents dataframes for training and testing

    PARAMETERS :
    -----------
        - dataframe : DataFrame
            Initial DataFrame to be split in two parts

        - frac : tuple
            A tuple that contain the proportion of train and test data to perform
            The first element correspond to the proportion of training set and the second to the test

    RETURN :
    ------
        - DataFrame , DataFrame : a tuple of DataFrame. The first element is the training and the second the test
    """

    # shuffel the datas
    dataframe = dataframe.sample(frac = 1)

    # Define a 70/30 training/testing split
    train_len = int(dataframe.shape[0]*frac[0])

    # Split the data in two parts 
    train_data = dataframe[:train_len].reset_index(drop=True)
    test_data = dataframe[train_len:].reset_index(drop=True)

    # Reset indexes
    train_data.reset_index(drop=True , inplace =True )
    test_data.reset_index(drop=True , inplace =True )

    return train_data, test_data


def generate_datas(classes = CLASSES, features =FEATURES, n_samples = N_SAMPLE, params = classes_params):
    """
    DESCRIPTION :
    ------------
        Generate synthetic data for every classes based on given features and parameters.

    PARAMETERS :
    -----------
        - classes : str
            The class for which data is generated.

        - features : list[str])
            List of features to generate data for (e.g:"height", "weight", "lifetime" ,"bark_days", "radius", "mass", "temperature", "income" . . .)

        - n_samples : int
            Number of samples to generate for each feature.

        - params : dict
            Dictionary containing parameters for each class and its features.

    RETURNS :
    --------
        - df : pandas.DataFrame
            A DataFrame containing the generated synthetic data.
            The DataFrame will have columns for each feature and an additional column for the class.
    """
    
    df = pd.DataFrame()
    
    #we iterate on every classes
    for classe , n_sample in zip(classes , n_samples) :
        sub_df = pd.DataFrame()
        # iterate over all features
        for feature in features:
            # find the name corresponding to the exact distribution of the feature
            distrib_name = get_distribution( params[classe][feature].__repr__() )
            # build the data according to the distribution find
            match distrib_name:
                case "gaussian" :
                    sub_df[feature] = gaussian_generator(params[classe][feature].mu, params[classe][feature].sigma, n_sample)
                    
                case "binomial":
                    sub_df[feature] = binomial_generator(params[classe][feature].n, params[classe][feature].p, n_sample)

                case "uniform":
                    sub_df[feature] = uniform_generator(params[classe][feature].a, params[classe][feature].b, n_sample)  

                case "poisson" :
                    sub_df[feature] = poisson_generator(params[classe][feature].lamda , n_sample )
    
        sub_df["class"] = classe

        # Concatenate all classes into a single dataframe
        df = pd.concat([df , sub_df]).reset_index(drop=True)
    
    # Shuffle the data
    df = df.sample(frac = 1)

    # reset the index
    df.reset_index(drop=True, inplace = True)

    return df