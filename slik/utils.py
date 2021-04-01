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
                             

def print_devider(title):
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))

def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)


def store_attribute(dict_file,output_path):    
    with open(f'{output_path}/store_file.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)


def store_pipeline(pipeline_object, pipeline_path):
    # save the model to disk
    pickle.dump(pipeline_object, open(pipeline_path, 'wb'))

def get_scores(y_true, y_pred):
    
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull,'w')
        
    def __exit__(self, exc_type, exc_va, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
