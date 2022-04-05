import pickle,yaml,os,pathlib,sys
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)
from .messages import log


def print_divider(title):

    """
    Expand print function with a clear differentiator using -.

    Parameters:
        Title: the title of the print statement
        
    Returns:
        None
    """
    log('\n{} {} {}\n'.format('-' * 15, title, '-' * 15))

def load_pickle(fp):

    """
    Load pickle file(data, model or pipeline object).

    Parameters:
        fp: the file path of the pickle files.
        
    Returns:
        Loaded pickle file

    """
    with open(fp, 'rb') as f:
        return pickle.load(f)


def store_attribute(dict_file,output_path):  
    """
    Store attributes of a dataframe as a dict.

    Parameters:
        dict_file: the dictionary.
        output_path: the path where the file is saved
        
    Returns:
        None
    """  
    with open(f'{output_path}/store_file.yaml', 'w') as file:
        yaml.dump(dict_file, file)


def store_pipeline(pipeline_object, pipeline_path):
    """
    Store the column transformer pipeline object.

    Parameters:
        pipeline_object: the pipeline object.
        pipeline_path: the path where the pipeline is saved
        
    Returns:
        None
    """
    # save the model to disk
    
    pickle.dump(pipeline_object, open(pipeline_path, 'wb'))

def get_scores(y_true, y_pred):

    """
    Get metrics of model performance such as accuracy, precision, recall and f1.

    Parameters:
        y_true: the target value of test/validation data.
        y_pred: the predicted value
        
    Returns:
        Accuracy, precision, recall and f1

    """
    
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }
    
def log_plot(args, plot_func, fp):

    '''
    Log the plots of your metrics and save output in a specified file path.

    Parameters:
        args: A tuple.
            Arguments required to plot the required metrics
        plot_func: A function
                Contains different method for plotting metrics such as
                ROC-AUC, PR-Curve
        fp: File Path
            The path to write the output logs of the plot
        
    Returns:
        None
    
    '''
    if not isinstance(args, (tuple)):
        args = (args,)

    plot_func(*args, fp)
    print(f'Logged {fp}')



class HiddenPrints:
    """
    Hide prints of a function

    Parameters:
        None
        
    Returns:
        None
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull,'w')
        
    def __exit__(self, exc_type, exc_va, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout