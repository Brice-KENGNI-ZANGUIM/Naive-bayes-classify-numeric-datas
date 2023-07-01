import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def print_confusion_matrix(label_true , label_predict, class_names, figsize = (8,7), cmap = plt.cm.Blues): 
    import itertools
    
    """
    DESCRIPTION : 
    ------------
        Affiche la transposée de la  matrice de confusion tel que fournie par 
        la fonction sklearn.metrics.confusion_matrix, sous forme de heatmap
    
    PARAMETERS :
    -----------
        label_true , label_predict : Array_type
             labels exactes et label prédit par le modèle
        class_name : list or Array_type
            label à attribuer aux observations : 0 - 1 , oui - non , bon - mauvais , ok - pas ok
    
    RETURN : None
    -------
    
    """    

    confusion_matrix = metrics.confusion_matrix( label_true , label_predict)
    print( confusion_matrix)
    #confusion_matrix = confusion_matrix.T
    with plt.style.context(('ggplot', 'seaborn')):
        fig = plt.figure(figsize=figsize, num=1)
        plt.imshow(confusion_matrix, interpolation='nearest',cmap= cmap )
        tick = [i for i in range(len(class_names))]
        plt.xticks( tick,class_names , )
        plt.yticks(tick ,class_names)
        plt.xlabel('\nPredict Label' , size = 3*np.min(figsize) )
        plt.ylabel('True Label' , size = 3*np.min(figsize))
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
                    plt.text(j, i,confusion_matrix[i, j], horizontalalignment="center",color="red" , fontsize = 2.3*np.min(figsize))
        plt.grid(None)
        plt.title('Confusion Matrix\n' , fontsize = 4*np.min(figsize))
        plt.colorbar() 


def plot_gaussian_distributions( gaussian ):
    """
    DESCRIPTION :
    ------------
        plot the histogram of gaussians distributes samples

    PARAMETERS :
    -----------
        - gaussian : list/array or tuple
            content a list of many samples datas to be plot as histogram
            ( sample_1 , sample_2 , . . . )

    RETURN : None
    -------
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for i in range(len(gaussian)) :
        ax.hist(gaussian[i], alpha=0.5, label=f"gaussian_{i}", bins= len(gaussian[i])//100 )

    ax.set_title("Histograms of Gaussian distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()
    
    
def plot_binomial_distributions(binomial):
    """
    DESCRIPTION :
    ------------
        plot the histogram of binomial distributes samples

    PARAMETERS :
    -----------
        - binomial : list/array or tuple
            content a list of many samples datas to be plot as histogram
            ( sample_1 , sample_2 , . . . )

    RETURN : None
    -------
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for i in range( len(binomial)) :
        ax.hist(binomial[i], alpha=0.5, label="binomial_{i}")

    ax.set_title("Histograms of Binomial distributions")
    ax.set_xlabel("Values")
    ax.set_ylabel("Frequencies")
    ax.legend()
    plt.show()
