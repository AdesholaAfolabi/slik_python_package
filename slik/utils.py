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
        documents = yaml.dump(dict_file, file)

def store_model(alg,model_path):
    # save the model to disk
    pickle.dump(alg, open(model_path, 'wb'))

    # save the pipeline to disk
    pipeline = file_object[1]
    pickle.dump(pipeline, open(pipeline_path, 'wb'))

if os.path.exists("./data/store_file.yaml"):
    config = yaml.safe_load(open("./data/store_file.yaml"))
    numerical_attribute = config['num_feat']
    categorical_attribute = config['lower_cat']
    hash_features = config['hash_feat']
    input_columns = config['input_columns']

else:
    pass