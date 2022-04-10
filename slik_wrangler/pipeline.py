"""Build Data and Model pipelines efficiently."""

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
import sklearn,warnings
import re
import warnings

# To display multiple output from a single cell.
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
from .loadfile import read_file
from .preprocessing import identify_columns,preprocess,map_target
from .utils import load_pickle,print_divider,store_pipeline, HiddenPrints, get_scores, log_plot
from slik_wrangler import plot_funcs as pf

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer    # to transform column of different types
from sklearn.model_selection import train_test_split            
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV    # to find best hyper parameters
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

class DenseTransformer(TransformerMixin):

    """

    Transform sparse matrix to a dense matrix.

    """

    def fit(self, X, y=None, **fit_params):
        """Fit a sparse matrix.

        Fit a sparse matrix with the DenseTransformer class

        Parameters
        ----------
        X: numpy array
            Sparse matrix to be fitted
        y: numpy array
            Target array
        """
        return self

    def transform(self, X, y=None, **fit_params):
        """Transform a fitted sparse matrix to a dense matrix.

        DenseTransformer tranforms a sparse matrix to a dense matrix.
        Some Transformer class do not work with sparse martix, hence
        the transformation.

        Parameters
        ----------
        X: numpy array
            Sparse matrix to be fitted
        y: numpy array
            Target array
        
        Returns
        -------
        Output: numpy array
            Dense matrix 
        """
        return X.todense()
    

def _build_model(dataframe=None,target_column=None,numerical_transformer=None,categorical_transformer=None
                ,pca=False, algorithm=None, balance_data=False,display_inline=True,
                grid_search=False,params=None,hashing=False,hash_size=500,project_path=None,
                **kwargs):
    
#     algorithm = algorithm.copy()
    data_path = os.path.join(project_path, 'data/')
    try:
        os.mkdir(data_path)
    except:
        pass
    
    identify_columns(dataframe,target_column,project_path=data_path,display_inline=display_inline,**kwargs)
    
    model_preprocessor_pipeline = os.path.join(project_path, 'model/')
    try:
        os.mkdir(model_preprocessor_pipeline)
    except:
        pass

    if os.path.exists(f"{project_path}/data/metadata/store_file.yaml"):
        config = yaml.safe_load(open(f"{project_path}/data/metadata/store_file.yaml"))
        numerical_attribute = config['num_feat']
        categorical_attribute = config['cat_feat']
        lower_categorical_attribute = config['lower_cat']
        hash_features = config['high_card_feat']
        input_columns = config['input_columns']
    else:
        raise ValueError('path: No file found in f"{project_path}/data/metadata/"')
    
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
    y = dataframe[f'{target_column}'].loc[list(train_df.index)]
    
    if balance_data:
        oversample = SMOTE()
        data_transformer.fit(train_df)
        encoder = data_transformer.transform(train_df)
        X, y = oversample.fit_resample(encoder,y)
        train_df = X
        X_train_copy = encoder
        X_train, X_test, y_train, y_test = train_test_split(train_df, y,\
                                                            stratify=y, test_size=0.20, random_state=0)
        
    else:
    
        X_train, X_test, y_train, y_test = train_test_split(train_df, y,\
                                                            stratify=y, test_size=0.20, random_state=0)

        X_train_copy = X_train.copy()
        data_transformer.fit(X_train_copy)
        X_train_copy = data_transformer.transform(X_train_copy)
    
         
    if pca:
        print_divider('Applying PCA to the data')
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
    
    exclusive_keyword = ['model','fit','hyperparameters']
    
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
                    raise ValueError(f'params:Only one of parameters {exclusive_keyword} should be set')
    
    if grid_search:
        # We can utilize params grid to check for best hyperparameters or transformers
        # The syntax here is pipeline_step_name__parameters and we need to chain if we have nested pipelines 
        parameters_grid = {}
        for key,value in params.items():
            parameters_grid[f'model__{key}'] = value
                # Doing a Grid Search
        grid_search = GridSearchCV(classifier, param_grid=parameters_grid)
        # fitting on our dataset
        grid_search.fit(X_train, y_train)  # Semicolon to not print estimator in notebook
        # set config to diagram for visualizing the pipelines/composite estimator
            
        
        model_path = os.path.join(model_preprocessor_pipeline,f'{algorithm.__class__.__name__}.pkl')
        store_pipeline(grid_search.best_estimator_,model_path)
            
        set_config(display='diagram')
        # Lets visualize the best estimator from grid search.
        output = grid_search.best_estimator_
        # saving pipeline as html format
        with open(f'{model_preprocessor_pipeline}/titanic_data_pipeline_estimator.html', 'w') as f:  
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
            
        model_path = os.path.join(model_preprocessor_pipeline,f'{algorithm.__class__.__name__}.pkl')
        store_pipeline(classifier,model_path)
        
#         set config to diagram for visualizing the pipelines/composite estimators
        set_config(display='diagram')
        output = classifier
        
        with open(f'{model_preprocessor_pipeline}/titanic_data_pipeline_estimator.html', 'w') as f:  
            f.write(estimator_html_repr(classifier))
        
            
#     X_test = data_transformer.transform(X_test)        
    y_pred = output.predict(X_test)
    
    print_divider('Metric Performance')
            
    met_perf = get_scores(y_test,y_pred)
    
    print(f'\nMetric performance on test data\n{met_perf}\n\n')
    print('\nconfusion matrix')
    
    print(confusion_matrix(y_test, y_pred))
    
    return output


    
def build_data_pipeline(data=None,target_column=None,id_column=None,clean_data=True,
                        project_path=None,numerical_transformer=None,categorical_transformer=None,
                        select_columns=None,pca=True,algorithm=None,grid_search=False,display_inline=False,
                        hashing=False,params=None,hash_size=500,balance_data=False,
                        **kwargs):
    """
    Build data and model pipeline.
    
    Build production ready pipelines efficiently. Specify numerical and categorical transformer.
    Function also helps to clean your data, reduce dimensionality and handle sparse categorical 
    features. 
    
    Parameters
    ------------
    data: str/ pandas dataframe
        Data path or Pandas dataframe.
        
    target_column: str
        target column name
        
    id_column: str
        id column name
        
    clean_data: Bool, default is True
        handle missing value, outlier treatment, feature engineering
        
    project_path: str/file path
        file path to processed data
        
    numerical_transformer: sklearn pipeline
        numerical transformer to transform numerical attributes
        
    categorical_transformer: sklearn pipeline
        categorical transformer to transform numerical attributes
        
    select_columns: list
        columns to be passed/loaded as a dataframe 
        
    pca: Bool, default is True
        reduce feature dimensionality
        
    algorithm: Default is None
        sklearn estimator
        
    grid_search: Bool. default is False
        select best parameter after hyperparameter tuning
        
    hashing: Bool. default is False
        handle sparse categorical features
        
    params: dict.
        dictionary of keyword arguments.  
        
    display_inline: Bool, default is True
        display dataframe print statement
        
    hash_size: int, default is 500
        size for hashing 
         
    Returns
    ---------
    Output:
        sklearn pipeline estimator

    """
    if project_path is None:
        raise ValueError("project_path: Expecting a file path and got None")
        
    data_path = os.path.join(project_path, 'data')
    
    try:
        os.mkdir(data_path)
    except:
        pass
    
    PROCESSED_TRAIN_PATH = os.path.join(data_path, 'train_data.pkl')
    
    if os.path.exists(PROCESSED_TRAIN_PATH):
        print_divider(f'Loading clean data')
        print(f'\nClean data exists in {PROCESSED_TRAIN_PATH}\n')
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
#         print(train_df.head())
    else:
        if data is None:
            raise ValueError("data: Expecting a pandas dataFrame or a data path, got None")
            
    if clean_data is True and data is None:
        raise ValueError("data: Expecting a pandas dataFrame or a data path, if clean_data set to True")
        
    if numerical_transformer is None:
        raise ValueError("numerical_transformer: Expecting a pipeline object for numerical transformation , got None")
        
    if categorical_transformer is None:
        raise ValueError("categorical_transformer: Expecting a pipeline object for numerical transformation , got None")
    
    if grid_search is True and params is None:
        raise ValueError("param grid: Expecting a dictionary of the parameters , got None")
        
    if algorithm is None:
        raise ValueError("algorithm: Expecting a sklearn algorithm , got None")
        
    if os.path.exists(project_path):
        pass
    else:
        os.mkdir(project_path)
        
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
    
    if not isinstance(id_column,str):
        errstr = f'The given type for id_column is {type(id_column).__name__}. Expected type is str'
        raise TypeError(errstr)
          
    target_column = f'transformed_{target_column}'
    
    if os.path.exists(PROCESSED_TRAIN_PATH):
        Question = input("Use clean data:(y/n) \n")
        if Question == ("y"):
            clean_data = False
            
        elif Question == ("n"):
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    train_df = data
                    if display_inline == True:
                        print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(train_df[target_column].unique())}')
                    if select_columns:
                        train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
                else:
                    train_df = read_file(data, input_col= select_columns, **kwargs)
                    if display_inline == True:
                        print(f'\nTarget column is {target_column}. Attribute in target column incldes:\n{list(train_df[target_column].unique())}')
            else:
                raise ValueError("data: Expecting a pandas dataFrame or a data path")

        
    if clean_data:
        if display_inline == False:
            logging = 'silent'
        preprocess(data=train_df,target_column=target_column,train=True,display_inline=display_inline,
                           project_path=project_path,select_columns=select_columns,logging=logging,**kwargs)
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
        

    else:
        if data is not None:
            train_df = data
            warnings.warn("Using raw data without proper data preprocessing can lead to several errors when building your model. It is advised tht you Set clean_data to True")
    
    output = _build_model(dataframe = train_df,target_column=target_column,numerical_transformer=numerical_transformer,\
                        categorical_transformer=categorical_transformer,pca=pca,algorithm=algorithm,\
                    grid_search=grid_search,params=params,id_column=id_column,display_inline=display_inline,hashing=hashing,
                        hash_size=hash_size,project_path=project_path,balance_data=balance_data)
    
    return output
    

def pipeline_transform_predict(data=None,select_columns=None,project_path=None,model_path=None):

    """
    Transform pipeline object and return Predictions.

    Transform dataframe based on slik build data pipeline function. Invoke model on 
    transformed data and return predictions
        
    Parameters
    -----------
    data: str/ pandas dataframe
        Data path or Pandas dataframe.
            
    select_columns: list
        columns to be passed/loaded as a dataframe
        
    project_path: str/file path
        path to project
        
    model_path: str/file path
        file path to model object
         
    Returns
    --------- 
    results: numpy array
        list of numpy array predictions

    """
    
    if os.path.exists(f"{project_path}/data/metadata/store_file.yaml"):
        config = yaml.safe_load(open(f"{project_path}/data/metadata/store_file.yaml"))
        input_columns = config['input_columns']
    
    else:
        raise ValueError('path: No file found in f"{project_path}/data/metadata/"')
        
    if select_columns: 
        for columns in select_columns:
            if columns in input_columns:
                pass
            else:
                raise ValueError(f"{columns} is not present in the training data.")
                
    with HiddenPrints():
        preprocess(data=data,train=False,display_inline=False,project_path=project_path,select_columns=input_columns)

    
                
    data = load_pickle(f'{project_path}/data/validation_data.pkl')
    data = data.reindex(columns = input_columns)
    MODEL = load_pickle(model_path)
    results = MODEL.predict(data)
    
    return results


def get_feature_names(column_transformer):

    """
    Get feature names after using column transformer object.

    Get feature names after trabsformations from each transformers 
    object in the column transformer class.

    Parameters
    ---------
    column_transformer: sklearn column transformer
    
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names
    
    
def evaluate_model(model_path=None,eval_data=None,select_columns=None,project_path=None,**kwargs):

    """
    Check model strength by validating model with an evaluation data.

    Evaluate model based on slik build data pipeline function. Invoke model on 
    transformed data and return evaluation plots in a file path.
    
    Parameters
    ------------
    model_path: str/file path
        file path to model object
        
    eval_data: str/ pandas dataframe
        Data path or Pandas dataframe.
        
    select_columns: list
        columns to be passed/loaded as a dataframe
    
    project_path: str/file path
        path to project        

    """
    if os.path.exists(f"{project_path}/data/metadata/store_file.yaml"):
        config = yaml.safe_load(open(f"{project_path}/data/metadata/store_file.yaml"))
        target_column = config['target_column']
        input_columns = config['input_columns']
        
    else:
        raise ValueError(f"{project_path}/data/metadata/store_file.yaml not found")
        
    target_column = target_column.replace('transformed_','')
    
    if eval_data is not None:
        if isinstance(eval_data, pd.DataFrame):
            eval_data = eval_data
            if select_columns:
                eval_data = manage_columns(eval_data,columns=select_columns,select_columns=True)
        else:
            eval_data = read_file(eval_data, input_col= select_columns,**kwargs)
        
    with HiddenPrints():    
        preprocess(data=eval_data,train=False,display_inline=False,project_path=project_path,select_columns=select_columns)
        data = load_pickle(f'{project_path}/data/validation_data.pkl')
        eval_data = map_target(eval_data,target_column)
        y_test = eval_data['transformed_'+target_column].loc[list(data.index)]
        data = data.reindex(columns = input_columns)
        estimator = load_pickle(model_path)
        
    y_pred = estimator.predict(data)
    y_proba = estimator.predict_proba(data)[:, 1]
    scores_valid = get_scores(y_test,y_pred)
    print_divider('Metric Performance')
    print(scores_valid)
    
    plot_path = os.path.join(project_path, 'plots/')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    from collections import defaultdict
    scores = defaultdict(int)
    print_divider('Saving Plots')
    # record scores
    for k, v in scores_valid.items():
        scores[k] += v 
    log_plot(scores, pf.scores, f'{plot_path}scores.png')

    # feature importance
    features = get_feature_names(estimator[0][0])
    feature_list = []
    for feat in features:
        prefix = re.search(r'(.*?)__', feat.lower()).group(0)
        attr = feat.replace(prefix,'')
        feature_list.append(attr)

    if not hasattr(estimator[1], 'get_feature_importance'):
        warnings.warn("Sklearn estimator (%s) does not "
                                 "provide get_feature_importance. "
                                 "Will make use of model coefficient as feature importance"
                                 %(type(estimator[1]).__name__))
            
    else:
        feature_importances_gain = estimator[1].get_feature_importance()
        log_plot((pd.Index(feature_list), feature_importances_gain, 'Feature Importance'),
                pf.feature_importance, f'{plot_path}feature_importance.png')

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    log_plot(cm, pf.confusion_matrix, f'{plot_path}confusion_matrix.png')

    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_pred)
    log_plot((fpr, tpr, roc_auc), pf.roc_curve, f'{plot_path}roc_curve.png')

    # precision-recall curve
    pre, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_pred)
    log_plot((pre, rec, pr_auc), pf.pr_curve, f'{plot_path}pr_curve.png')
    
