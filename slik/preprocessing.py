import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from IPython.display import display
import yaml
from numpy import percentile

import matplotlib.pyplot as plt
from .loadfile import read_file
from .utils import store_attribute,print_devider
from .plot_funcs import plot_nan
# from utils.config import TRAIN_PATH_CLICK,TRAIN_PATH_SMS,TEST_PATH,PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH

def get_attributes(data=None,target_column=None):
    
    '''
    Returns the categorical features and Numerical features in a data set
    Parameters:
    -----------
        data: DataFrame or named Series
            Data set to perform operation on.
        target_column: str
            Label or Target column
    Returns:
    -------
        List
            A list of all the categorical features and numerical features in a dataset.
    '''
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    num_attributes = data.select_dtypes(exclude=['object', 'datetime64']).columns.tolist()
    cat_attributes = data.select_dtypes(include=['object']).columns.tolist()
    
    if target_column:
        num_attributes.remove(target_column)
    return num_attributes, cat_attributes

def identify_columns(data=None,target_column=None, high_dim=100, verbose=True, save_output=True):
    
    """
        Identifies numerical attributes ,categorical attributes with sparse features and categorical attributes with lower features
        present in the data with an option to save them in a yaml file.
     Parameters:
    -----------
        data: DataFrame or named Series 
        target_column: str
            Label or Target column.
        high_dim: int, default 100
            Number to identify categorical attributes greater than 100 features
        verbose: Bool, default=True
            display print statement
        save_output: Bool, default = True
            save output in the data path.   
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    num_attributes, cat_attributes = get_attributes(data,target_column)
        
    low_cat = []
    hash_features = []
    dict_file = {}
    input_columns = [cols for cols in data.columns]
    input_columns.remove(target_column)
  
    for item in cat_attributes:
        if data[item].nunique() > high_dim:
            if verbose:
                print('\n {} has a high cardinality. It has {} unique attributes'.format(item, data[item].nunique()))
            hash_features.append(item)
        else:
            low_cat.append(item)
    if save_output:
        dict_file['num_feat'] = num_attributes
        dict_file['cat_feat'] = cat_attributes
        dict_file['hash_feat'] = hash_features
        dict_file['lower_cat'] = low_cat
        dict_file['input_columns'] = input_columns
        store_attribute(dict_file)
        
        print_devider('Saving Attributes in Yaml file')
        print('\nDone!. Data columns successfully identified and attributes are stored in data/')
        
def detect_outliers(dataframe=None,y=None,num_features=None,n=None,remove=True):
        
    '''
    Detect outliers present in the numerical features and includes an option of removal.
    Parameters:
    ------------------------
    data: DataFrame or name Series.
        Data set to perform operation on.
    num_features: List, Series, Array.
        Numerical features to perform operation on. If not provided, we automatically infer from the dataset.
    n: int.
        Values used in searching for outliers
    remove: bool, Default True
        Remove outliers detected in the data.

    Returns:
    -------
        Dataframe
            A new dataframe after removing outliers.
    
    '''
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    if not isinstance(y,str):
        errstr = f'The given type for target_column is {type(y).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    
    if n is None:
        n = 2
    
    outlier_indices = []
    
    df = data.copy()
    
    if num_features is None:
        num_attributes, cat_attributes = get_attributes(data,y)
    else:
        num_attributes = features

    for column in num_attributes:
        
        data.loc[:,column] = abs(data[column])
        mean = data[column].mean()

        #calculate the interquartlie range
        q25, q75 = np.percentile(data[column].dropna(), 25), np.percentile(data[column].dropna(), 75)
        iqr = q75 - q25

        #calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower,upper = q25 - cut_off, q75 + cut_off

        #identify outliers
        # Determine a list of indices of outliers for feature col
        outlier_list_col = data[(data[column] < lower) | (data[column] > upper)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)
        
        if remove:
            df.loc[:,column] = df[column].apply(lambda x : mean 
                                                        if x < lower or x > upper else x)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    print_devider('Table idenifying Outliers present')
    display(data.loc[multiple_outliers])

    return df


def age(dataframe=None, age_col=None):

    '''
    The age attribute is binned into 5 (baby/toddler, child, young adult, mid age and elderly).
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    age_col: the name of the age column in the dataset. A string is expected
        The column to perform the operation on.

    Returns
    -------
        Dataframe with binned age attribute:
            New age column called Age Group
    '''
    
    if dataframe is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(age_col,str):
        errstr = f'The given type for age_col is {type(age_col).__name__}. Expected type is string'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    
#     handle_nan(dataframe=dataframe)
    bin_labels = ['Toddler/Baby', 'Child', 'Young Adult', 'Mid-Age', 'Elderly']
    data['Age Group'] = pd.cut(data[age_col], bins = [0,2,17,30,45,99], labels = bin_labels)
    data['Age Group'] = data['Age Group'].astype(str)

    return data
    

# concatenating name and version to form a new single column
def concat_feat(data):
    data['gender_location'] = data['gender'] + data['location_state']
    data['os_name_version'] = data['os_name'] + data['os_version'].astype(str)
    data['age_bucket'] = data.apply(lambda x: _age(x['age']), axis=1)
#         data['interactions'] = data['gender'] + data['age_bucket']
    
def check_nan(dataframe=None, plot=False, verbose=True):
    
    '''
    Display missing values as a pandas dataframe.
    Parameters
    ----------
        data: DataFrame or named Series
        plot: bool, Default False
            Plots missing values in dataset as a heatmap
        verbose: bool, Default False
            
    
    Returns
    -------
        Matplotlib Figure:
            Heatmap plot of missing values
    '''
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    df['missing_percent'] = missing_percent
    nan_values = df.set_index('features')['missing_percent']
    
    print_devider('Count and Percentage of missing value')
    if plot:
        plot_nan(nan_values)
    if verbose:
        display(df)
        
    check_nan.df = df
    
def handle_cat_feat(data,fillna,cat_attr):
    if fillna == 'mode':
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(data[item].value_counts().index[0])
            
    else:
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(fillna)
    return data

def handle_nan(dataframe=None,target_name=None, strategy='mean',fillna='mode',\
               drop_outliers=False,thresh_y=None,thresh_x=None, verbose =False,
               **kwargs):
    
    """
    Fill missing values of categorical features and numerical features.
    Args:
    ------------------------
    data: DataFrame or name Series.
            Data set to perform operation on.
    target_name: str
            Name of the target column
    strategy: str
        Method of filling numerical features
    fillna: str
        Method of filling categorical features
    drop_outliers: bool, Default False
            Drops outliers present in the data.
    thresh_x: Int.
            Threshold for dropping rows with missing values 
    thresh_y: Int.
            Threshold for dropping columns with missing value   
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    check_nan(data,verbose=verbose)
    df = check_nan.df
    
    if thresh_x:
        thresh_x = thresh_x/100
        drop_row = data.shape[1] * thresh_x
        data = data.dropna(thresh=drop_row)
        
    if thresh_y:
        drop_col = df[df['missing_percent'] > thresh_y].features.to_list()
        print(f'\nMissing Columns with {thresh_y}% missing value : {drop_col}')
        data = drop_cols(data,drop_col)
        print(f'\nNew data shape is {data.shape}')
    
    if drop_outliers:
        if target_name is None:
            raise ValueError("target_name: Expecting a str for the target_name, got 'None'")
        data = detect_outliers(data,target_name,**kwargs)

    num_attributes, cat_attributes = get_attributes(data,target_name)

    if strategy == 'mean':
        for item in num_attributes:
            data.loc[:,item] = data[item].fillna(data[item].mean())
            
    elif strategy == 'median':
        for item in num_attributes:
            median = data[item].median()
            data.loc[:,item] = data[item].fillna(median)
            
    elif strategy == 'mode':
        for item in num_attributes:
            mode = data[item].mode()[0]
            data.loc[:,item] = data[item].fillna(mode)
   
    else:
        raise ValueError("method: must specify a fill method, one of [mean, mode or median]'")
        
    data = handle_cat_feat(data,fillna,cat_attributes)
            
    return data

def drop_cols(dataframe=None,features=None):
    
    '''
    Drop features from a pandas dataframe.
    Parameters
    ----------
        data: DataFrame or named Series
        features: list of features you want to drop
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe without the dropped features
    '''
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if features is None:
        raise ValueError("data: Expecting a list, got 'None'")
        
    data = dataframe.copy()
    data = data.drop(features,axis=1)
    return data
    
def map_column(data=None,column_name=None,items=None):
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(column_name,str):
        errstr = f'The given type for column_name is {type(column_name).__name__}. Expected type is str'
        raise TypeError(errstr)
  
    if not isinstance(items,dict):
        errstr = f'The given type for items is {type(items).__name__}. Expected type is dict'
        raise TypeError(errstr)

    data.loc[:,column_name] = data[column_name].map(items)
    
    
# def preprocess(dataframe=None,train=True,validation_path=None):
#     if train:
#         dataframe = dataframe[~dataframe['customer_class'].isnull()]

#         map_target(dataframe,'event_type')
#         dataframe = handle_nan(dataframe,fillna='missing',drop_outliers=True)
#         dataframe = drop_cols(dataframe,columns=['msisdn.1'])
#         dataframe.to_pickle(PROCESSED_TRAIN_PATH)
#         print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}')
        
#     else:
#         data = raw.read_data(path=validation_path)
#         map_target(data,'event_type')
#         data = handle_nan(data,fillna='missing',drop_outliers=True)
#         data = drop_cols(data,columns=['msisdn.1'])

#         data.to_pickle(PROCESSED_TEST_PATH)

#         print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TEST_PATH}')
    