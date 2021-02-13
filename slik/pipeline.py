from .preprocessing import preprocess,identify_columns,manage_columns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
import os
from .loadfile import read_file
from .utils import load_pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer                   # to transform column of different types
from sklearn.model_selection import train_test_split            
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV             # to find best hyper parameters
from sklearn import set_config                      # to change the display
from sklearn.utils import estimator_html_repr       # to save the diagram into HTML format
            

def build_model(dataframe=None,target_column=None,numerical_tranformer=None,categorical_transformer=None,pca=True,algorithm=None):
    
    numerical_features ,categorical_features = pp.get_attributes(dataframe)
    
    data_transformer = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)])
    
    X_train, X_test, y_train, y_test = train_test_split(dataframe, target_column,\
                                                        stratify=target_column, test_size=0.25, random_state=0)
    X_train_copy = X_train.copy()
    X_train_copy = data_transformer.fit_transform(X_train_copy)
    if pca:
        pca_ = PCA().fit(X_train_copy)
        pca_evr = pca_.explained_variance_ratio_
        cumsum_ = np.cumsum(pca_evr)
        dim_95 = np.argmax(cumsum_ >= 0.95) + 1
        instances_, dims_ =  X_train_copy.shape
        dim_reduction = PCA(dim_95)
        print(f"Dimension is reduced from {dims_} to {dim_95} while retaining 95% of variance.")
    else:
        dim_reduction = 'passthrough'
    
    preprocessor = Pipeline(steps=[('data_transformer', data_transformer),\
                                   ('reduce_dim',dim_reduction)])
    classifier = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=0, max_iter=10000))])
    
    
    # We can utilize params grid to check for best hyperparameters or transformers
    # The syntax here is pipeline_step_name__parameters and we need to chain if we have nested pipelines 
    param_grid = {
        'classifier__C': [0.1, 1.0, 10, 100],
    'classifier__solver': ['liblinear','newton-cg']
    }

    # Doing a Grid Search
    grid_search = GridSearchCV(classifier, param_grid=param_grid)

    # fitting on our dataset
    grid_search.fit(X_train, y_train)  # Semicolon to not print estimator in notebook

    # set config to diagram for visualizing the pipelines/composite estimators
    set_config(display='diagram')

    # Lets visualize the best estimator from grid search.
    grid_search.best_estimator_

    # saving pipeline as html format
    with open('titanic_data_pipeline_estimator.html', 'w') as f:  
        f.write(estimator_html_repr(grid_search.best_estimator_))
    return grid_search    

    
def build_data_pipeline(dataframe=None,target_column=None,id_column=None,clean_data=True,
                        data_path=None,verbose=True,processed_data_path=None,
                       select_columns=None,**kwargs):
    """

     Function contains the pipeline methods to be used.
     It is broken down into numerical, categorical and hash pipeline

    """
    if dataframe is None and data_path is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series or a data path, got None")
        
    if processed_data_path is None:
        raise ValueError("dataframe: Expecting a data path and got None")
        
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
                           processed_data_path=processed_data_path,**kwargs)
        
        train_df = load_pickle(PROCESSED_TRAIN_PATH)
        if select_columns:
            train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
        prefix_name = f'transformed_{target_column}'
        
        identify_columns(train_df,prefix_name,id_column,verbose=verbose)
    
    
#      train_df
#         if scaling:
            
#             self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', StandardScaler())])
#             self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                        ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
#             self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                   ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
            
#         elif normalization:
            
#             self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
#             self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                        ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
#             self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                   ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
            
#         else:
            
#             self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                        ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
#             self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
#                                   ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
    
def pipeline(hash_size):
        
    """
    
        Function contains the pipeline methods to be used.
        It is broken down into numerical, categorical and hash pipeline
            
    """
    num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                    ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
    hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
    
    return num_pipeline,cat_pipeline,hash_pipeline

def train_test(data,hash_size,test_size):
    identify_columns(data,high_dim=hash_size, verbose=True)
    y = data['event_type']
    X = data.drop(['event_type'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,stratify = y,test_size=test_size)
    return X_train, X_test, y_train, y_test

def fit_transform(data, hash_size, test_size):
    
    """
    
    Function that builds the pipeline and returns the 
    pipeline object and the data to be used for modeling
            
    Args:
        hash_bucket size
    
    Returns:
        pipeline object
        data to be used for training after being transformed by the pipeline
    
    """

    num_pipeline,cat_pipeline,hash_pipeline = pipeline(hash_size)
    full_pipeline = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_attribute),
        ('cat', cat_pipeline, categorical_attribute),
        ('hash', hash_pipeline, hash_features)
    ])
    X_train, X_test, y_train, y_test = train_test(data,hash_size,test_size)
    
    full_pipeline.fit(X_train)
    
    X_train = full_pipeline.transform(X_train)
    X_test = full_pipeline.transform(X_test)
    
    print(X_train.shape)
    return X_train, X_test, y_train, y_test, full_pipeline
#     def build_object(self, hash_size, scaling = None, normalization =None):
        
#         """
        
#         Function that builds the pipeline and returns the 
#         pipeline object and the data to be used for modeling
                
#         Args:
#             hash_bucket size, scaling, normalization
        
#         Returns:
#             pipeline object
#             data to be used for training after being transformed by the pipeline
        
#         """
        
#         self.identify_cat()
#         self.pipeline(hash_size, scaling, normalization)
#         self.full_pipeline = ColumnTransformer(
#         transformers=[
#             ('num', self.num_pipeline, self.num_attributes),
#             ('cat', self.cat_pipeline, self.low_cat),
#             ('hash', self.hash_pipeline, self.hash_features)
#         ])
        
#         self.X = self.data
        
#         self.full_pipeline.fit(self.X)
        
#         self.X = self.full_pipeline.transform(self.X)
        
#         print(self.X.shape)
#         return self.X, self.full_pipeline
        


