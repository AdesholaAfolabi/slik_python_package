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
import os,scipy,inspect

# To display multiple output from a single cell.
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
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
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             roc_curve,
                             roc_auc_score,
                             precision_recall_curve,
                             average_precision_score)


def get_scores(y_true, y_pred):
    return {
      'accuracy': accuracy_score(y_true, y_pred),
      'precision': precision_score(y_true, y_pred),
      'recall': recall_score(y_true, y_pred),
      'f1': f1_score(y_true, y_pred),
    }


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
    

def build_model(dataframe=None,target_column=None,numerical_transformer=None,categorical_transformer=None
                ,pca=True,algorithm=None,grid_search=False,params=None,hashing=False,hash_size=500,
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
        use_cols =  numerical_attribute + categorical_attribute + hash_features
        
    else:
        data_transformer = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_attribute),
            ('categorical', categorical_transformer, categorical_attribute)])
        use_cols =  numerical_attribute + categorical_attribute
    
    train_df = manage_columns(dataframe,columns=input_columns,select_columns=True)
    y = dataframe[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(train_df, y,\
                                                        stratify=y, test_size=0.25, random_state=0)
    
    X_train_copy = X_train.copy()
    data_transformer.fit(X_train_copy)
    X_train_copy = data_transformer.transform(X_train_copy)
    
    if pca:
        print_devider('Applying PCA to the data')
        if scipy.sparse.issparse(X_train_copy):
            X_train_array = X_train_copy.toarray()
#         elif isinstance(X_train_copy, np.ndarray):
#             X_train_array = X_train_copy
        else:
            X_train_array = X_train_copy
#         X_train_copy = data_transformer.fit_transform(X_train_copy)
        pca_ = PCA().fit(X_train_array)
        pca_evr = pca_.explained_variance_ratio_
        cumsum_ = np.cumsum(pca_evr)
        dim_95 = np.argmax(cumsum_ >= 0.95) + 1
        instances_, dims_ =  X_train_copy.shape
        dim_reduction = PCA(dim_95)
        print(f"\nDimension reduced from {dims_} to {dim_95} while retaining 95% of variance.")
        if hashing:
            preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                       ('to_dense', DenseTransformer()),\
                                       ('reduce_dim',dim_reduction)])
        else:
            preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                       ('reduce_dim',dim_reduction)])
    else:
        if hashing:
            preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                   ('to_dense', DenseTransformer())])
        else:
            preprocessor = Pipeline(steps=[('data_transformer', data_transformer)])
    
    classifier = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', algorithm)])
    
    if params:
        model_params = {}
        fit_params = {}
        hyperparamers_params = {}
        keys = [key for key in params.keys()]
        for first_key in keys:
            for key,value in params[first_key].items():
                key = f'model__{key}'
                if first_key == 'model':
                    model_params[key] = value
                elif first_key == 'fit':
                    fit_params[key] = value 
                elif first_key == 'hyperparameters':
                    hyperparamers_params [key] = value
                else:
                    raise ValueError("params:'Only one of parameters {} should be set'.format(exclusive_keyword)")
    
    if grid_search:
        # We can utilize params grid to check for best hyperparameters or transformers
        # The syntax here is pipeline_step_name__parameters and we need to chain if we have nested pipelines 
        parameters_grid = {}
        for key,value in param_grid.items():
            parameters_grid[f'model__{key}'] = value
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
        if params:
            kwargsList = inspect.getfullargspec(algorithm.fit)[0]
            if len(fit_params)>0:
                
    #             X_val = data_transformer.transform(X_test) 
    #             if 'eval_set' in kwargsList:
    #                 fit_params['model__eval_set'] =  (X_test, y_test)
                try:
                    classifier.set_params(**model_params)
                    classifier.fit(X_train, y_train,**fit_params)
                except:
                    if 'cat_features' in kwargsList:
                        cate_features_index = [X_train.columns.get_loc(col) for col in X_train.columns][len(numerical_attribute):]
                    fit_params['model__cat_features'] = cate_features_index
                    classifier.set_params(**model_params)
                    classifier.fit(X_train, y_train,**fit_params)
            else:
                try:
                    classifier.set_params(**model_params)
                    classifier.fit(X_train, y_train) 
                except:
                    if 'cat_features' in kwargsList:
                        cate_features_index = [X_train.columns.get_loc(col) for col in X_train.columns][len(numerical_attribute):]
                    fit_params['model__cat_features'] = cate_features_index
                    classifier.set_params(**model_params)
                    classifier.fit(X_train, y_train,**fit_params) 
        else:
            classifier.set_params(**model_params)
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
            
#     X_test = data_transformer.transform(X_test)        
    y_pred = output.predict(X_test)
    
    print_devider('Metric Performance')
            
    met_perf = get_scores(y_test,y_pred)
    print(f'\nMetric performance on test data\n{met_perf}\n\n')

    return output


    
def build_data_pipeline(data=None,target_column=None,id_column=None,clean_data=True,
                        verbose=True,processed_data_path=None,
                        numerical_transformer=None,categorical_transformer=None,
                       select_columns=None,pca=True,algorithm=None,grid_search=False,
                        hashing=False,params=None,hash_size=500,model_preprocessor_pipeline=None,
                        **kwargs):
    """

     Function contains the pipeline methods to be used.

    """
    
    if processed_data_path is None:
        raise ValueError("processed_data_path: Expecting a file path and got None")
        
    PROCESSED_TRAIN_PATH = os.path.join(processed_data_path, 'train_data.pkl')
    
    if os.path.exists(PROCESSED_TRAIN_PATH):
        print_devider(f'Loading clean data')
        print(f'\nClean data exists in {PROCESSED_TRAIN_PATH}\n')
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
        
    else:
        if data is None:
            raise ValueError("data: Expecting a DataFrame or Series or a data path, got None")
            
    if clean_data is True and data is None:
        raise ValueError("data: Expecting a DataFrame or Series or a data path, if clean_data set to True")
        
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
          

    if os.path.exists(PROCESSED_TRAIN_PATH):
        Question = input("Use clean data:(y/n) \n")
        if Question == ("y"):
            clean_data = False
            target_column = f'transformed_{target_column}'
        elif Question == ("n"):
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    train_df = data
                    print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(train_df[target_column].unique())}')
                    if select_columns:
                        train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
                else:
                    train_df = read_file(data, input_col= select_columns, **kwargs)
                    print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(train_df[target_column].unique())}')
        
        
    if clean_data:    
        preprocess(data=train_df,target_column=target_column,train=True,verbose=verbose,
                           processed_data_path=processed_data_path,select_columns=select_columns,**kwargs)
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
        target_column = f'transformed_{target_column}'
    
    output = build_model(dataframe = train_df,target_column=target_column,numerical_transformer=numerical_transformer,\
                        categorical_transformer=categorical_transformer,pca=pca,algorithm=algorithm,
                        grid_search=grid_search,params=params,id_column=id_column,verbose=verbose,hashing=hashing,
                        model_preprocessor_pipeline=model_preprocessor_pipeline,hash_size=hash_size)
    
    return output
    

def pipeline_transform(data=None,select_columns=None,processed_data_path=None):
    preprocess(data=data,train=False,verbose=False,processed_data_path=processed_data_path,\
                  task='classification',select_columns=select_columns)
    
    if os.path.exists("./data/store_file.yaml"):
        config = yaml.safe_load(open("./data/store_file.yaml"))
        input_columns = config['input_columns']
    
    else:
        raise ValueError("path: No file path /data/store_file.yaml found")
        
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

