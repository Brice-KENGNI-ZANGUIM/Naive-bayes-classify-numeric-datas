############################################################################################################
##########################                    MODULES IMPORTATION                 ##########################
############################################################################################################

import pprint
pp = pprint.PrettyPrinter()

from datas_generators import generate_datas, split_datas
from bayes_classifier import compute_training_params, predict_class
from configurations import  FEATURES, CLASSES
from plots import print_confusion_matrix



if __name__ == "__main__" :

    # Define the feature's names of data to generate
    datas = generate_datas()
    print( datas.head() )

    # split the datas into two parts : a train and a test data
    train_data , test_data = split_datas( dataframe= datas , frac = (.4, .6))
    print(train_data , test_data)

    # Estimate the features distribution's parameters and also the prior probability of classes 
    train_params, train_class_probs = compute_training_params(train_data)
    pp.pprint( train_class_probs )
    pp.pprint(train_params)

    # make predictions
    pred = predict_class( train_data.loc[0,FEATURES] , train_params, train_class_probs)
    print(pred)
    print(train_data.loc[0,"class"])

    # Plot confusion matrix 
    true_classes    = test_data["class"]
    predict_classes = test_data[FEATURES].apply( lambda x : predict_class(x , train_params, train_class_probs ) , axis=1 )
    print_confusion_matrix( true_classes , predict_classes , CLASSES)




