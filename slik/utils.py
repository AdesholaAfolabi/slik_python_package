import sys  
sys.path.insert(0, '/Users/terra-016/Downloads/mlflow-titanic/slik_python_package/')
import pickle,yaml,os

def print_devider(title):
    print('\n{} {} {}\n'.format('-' * 25, title, '-' * 25))

def load_pickle(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)

def store_attribute(dict_file):
    try:
        os.mkdir('data/')
    except:
        pass
    with open(r'./data/store_file.yaml', 'w') as file:
        yaml.dump(dict_file, file)

def store_pipeline(pipeline_object, pipeline_path):
    # save the model to disk
    
    pickle.dump(pipeline_object, open(pipeline_path, 'wb'))
        
    

#     # save the pipeline to disk
#     pipeline = file_object[1]
#     pickle.dump(pipeline, open(pipeline_path, 'wb'))

