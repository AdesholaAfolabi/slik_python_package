from .preprocessing import preprocess,identify_columns,manage_columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
import os

# To display multiple output from a single cell.
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

from .loadfile import read_file
from .preprocessing import identify_columns,preprocess
from .utils import load_pickle,print_devider,store_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer                   # to transform column of different types
from sklearn.model_selection import train_test_split            
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV             # to find best hyper parameters
from sklearn import set_config                      # to change the display
from sklearn.utils import estimator_html_repr       # to save the diagram into HTML format
import yaml
import numpy as np


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    

def build_model(dataframe=None,target_column=None,numerical_transformer=None,categorical_transformer=None
                ,pca=True,algorithm=None,grid_search=False,param_grid=None,hashing=False,hash_size=500,
                model_preprocessor_pipeline=None,**kwargs):
    
    identify_columns(dataframe,target_column,**kwargs)
   
    if os.path.exists("./data/store_file.yaml"):
        config = yaml.safe_load(open("./data/store_file.yaml"))
        numerical_attribute = config['num_feat']
        categorical_attribute = config['cat_feat']
        lower_categorical_attribute = config['lower_cat']
        hash_features = config['hash_feat']
        input_columns = config['input_columns']
    else:
        raise ValueError("path: No file path /data/store_file.yaml found")
    
    if hashing:
        hash_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
        categorical_attribute = lower_categorical_attribute
        data_transformer = ColumnTransformer(transformers=[
                                ('numerical', numerical_transformer, numerical_attribute),
                                ('categorical', categorical_transformer, categorical_attribute),
                                ('hasher', hash_transformer , hash_features)])
        
    else:
        data_transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_attribute),
            ('categorical', categorical_transformer, categorical_attribute)])
    
    train_df = manage_columns(dataframe,columns=input_columns,select_columns=True)
    y = dataframe[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(train_df, y,\
                                                        stratify=y, test_size=0.25, random_state=0)
    X_train_copy = X_train.copy()
    if pca:
        print_devider('Applying PCA to the data')
        X_train_copy = data_transformer.fit_transform(X_train_copy)
        pca_ = PCA().fit(X_train_copy.toarray())
        pca_evr = pca_.explained_variance_ratio_
        cumsum_ = np.cumsum(pca_evr)
        dim_95 = np.argmax(cumsum_ >= 0.95) + 1
        instances_, dims_ =  X_train_copy.shape
        dim_reduction = PCA(dim_95)
        print(f"\nDimension reduced from {dims_} to {dim_95} while retaining 95% of variance.")
        preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                   ('to_dense', DenseTransformer()),\
                                   ('reduce_dim',dim_reduction)])
    else:
        preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                   ('to_dense', DenseTransformer())])
    
    classifier = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', algorithm)])
   
    
    if grid_search:
        # We can utilize params grid to check for best hyperparameters or transformers
        # The syntax here is pipeline_step_name__parameters and we need to chain if we have nested pipelines 
        parameters_grid = {}
        for key,value in param_grid.items():
            parameters_grid[f'classifier__{key}'] = value
                # Doing a Grid Search
        grid_search = GridSearchCV(classifier, param_grid=parameters_grid)
        # fitting on our dataset
        grid_search.fit(X_train, y_train)  # Semicolon to not print estimator in notebook
        # set config to diagram for visualizing the pipelines/composite estimator
            
        if model_preprocessor_pipeline:
            model_path = os.path.join(model_preprocessor_pipeline,f'{algorithm.__class__.__name__}.pkl')
            store_pipeline(grid_search.best_estimator_,model_path)
            
        set_config(display='diagram')
        # Lets visualize the best estimator from grid search.
        output = grid_search.best_estimator_
        # saving pipeline as html format
        with open('titanic_data_pipeline_estimator.html', 'w') as f:  
            f.write(estimator_html_repr(grid_search.best_estimator_))
            
    else:
        classifier.fit(X_train, y_train)
        
        if model_preprocessor_pipeline:
            try:
                os.mkdir(model_preprocessor_pipeline)
            except:
                pass
            model_path = os.path.join(model_preprocessor_pipeline,f'{algorithm.__class__.__name__}.pkl')
            store_pipeline(classifier,model_path)
        
#         set config to diagram for visualizing the pipelines/composite estimators
        set_config(display='diagram')
        output = classifier
        with open('titanic_data_pipeline_estimator.html', 'w') as f:  
            f.write(estimator_html_repr(classifier))
            
        
            
    # After you train the model using fit(), save like this - 
        
    return output


    
def build_data_pipeline(dataframe=None,target_column=None,id_column=None,clean_data=True,
                        data_path=None,verbose=True,processed_data_path=None,
                        numerical_transformer=None,categorical_transformer=None,
                       select_columns=None,pca=True,algorithm=None,grid_search=False,
                        hashing=False,param_grid=None,hash_size=500,model_preprocessor_pipeline=None,
                        **kwargs):
    """

     Function contains the pipeline methods to be used.

    """
    if dataframe is None and data_path is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series or a data path, got None")
        
    if processed_data_path is None:
        raise ValueError("dataframe: Expecting a data path and got None")
        
    if numerical_transformer is None:
        raise ValueError("numerical_transformer: Expecting a pipeline object for numerical transformation , got None")
        
    if categorical_transformer is None:
        raise ValueError("categorical_transformer: Expecting a pipeline object for numerical transformation , got None")
    
    if grid_search is True and param_grid is None:
        raise ValueError("param grid: Expecting a dictionary of the parameters , got None")
        
    if algorithm is None:
        raise ValueError("algorithm: Expecting a sklearn algorithm , got None")
        
    if os.path.exists(processed_data_path):
        pass
    else:
        os.mkdir(processed_data_path)
        
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
    
    if not isinstance(id_column,str):
        errstr = f'The given type for id_column is {type(id_column).__name__}. Expected type is str'
        raise TypeError(errstr)
          
    PROCESSED_TRAIN_PATH = os.path.join(processed_data_path, 'train_data.pkl')
        
    
    if data_path:
        train_df = read_file(data_path, input_col= select_columns, **kwargs)
        print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(train_df[target_column].unique())}')
        
    else:
        train_df = dataframe
        print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(data[target_column].unique())}')
        if select_columns:
            train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
        identify_columns(train_df,target_column,verbose=verbose)
        
    if clean_data:    
        preprocess(dataframe=train_df,target_column=target_column,train=True,verbose=verbose,
                           processed_data_path=processed_data_path,select_columns=select_columns,**kwargs)
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
#     if select_columns:
#         train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
    prefix_name = f'transformed_{target_column}'
    
    output = build_model(dataframe = train_df,target_column=prefix_name,numerical_transformer=numerical_transformer,\
                        categorical_transformer=categorical_transformer,pca=pca,algorithm=algorithm,
                        grid_search=grid_search,param_grid=param_grid,id_column=id_column,verbose=verbose,hashing=hashing,
                        model_preprocessor_pipeline=model_preprocessor_pipeline)
    
    return output
    

def pipeline_transform(dataframe=None,data_path=None,select_columns=None,processed_data_path=None):
    preprocess(data_path=data_path,train=False,verbose=False,processed_data_path=processed_data_path,\
                  task='classification',select_columns=select_columns)
    
    if os.path.exists("./data/store_file.yaml"):
        config = yaml.safe_load(open("./data/store_file.yaml"))
        input_columns = config['input_columns']
        
    if select_columns: 
        for columns in select_columns:
            if columns in input_columns:
                pass
            else:
                raise ValueError(f"{column} is not present in the trained model.")
    data = load_pickle('./data/validation_data.pkl')
    data = data.reindex(columns = input_columns)
    return data
    

def train_test(dataframe,target_column,test_size,**kwargs):
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target_column ,stratify = target_column,test_size=test_size,
                                                       **kwargs)
    return X_train, X_test, y_train, y_test

