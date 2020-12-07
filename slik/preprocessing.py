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

def get_attributes(data=None,y=None):
    '''
    Returns the categorical features and Numerical features in a data set
    Parameters:
    -----------
        data: DataFrame or named Series 
        y: Label or Target.
    Returns:
    -------
        List
            A list of all the categorical features and numerical features in a dataset.
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    num_attributes = data.select_dtypes(exclude=['object', 'datetime64']).columns.tolist()
    cat_attributes = data.select_dtypes(include=['object']).columns.tolist()
    
    num_attributes.remove(y)
    return num_attributes, cat_attributes

def identify_columns(data=None,y=None, high_dim=100, verbose=True, save_output=True):
    
    """
        This funtion takes in the data, identify the numerical and categorical
        attributes and stores them in a list
        
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    num_attributes, cat_attributes = get_attributes(data,y)
        
    low_cat = []
    hash_features = []
    dict_file = {}
    input_columns = [cols for cols in data.columns]
    input_columns.remove(y)
  
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
        
def detect_outliers(data=None,y=None,features=None,n=None,remove=True):
        
    '''
    This function takes in the numerical data and removes outliers
    
    '''
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if n is None:
        n = 2
    
    outlier_indices = []
    
    df = data.copy()
    
    if features is None:
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


def _age(age):
    if  20 <= age <= 30:
        column = 'young'
    elif 30 < age <=50:
        column = 'middle_age'
    elif age>50:
        column = 'elder'
    else:
        column = 'missing'
        #raise ValueError(f'Invalid hour: {age}')
    return column

# concatenating name and version to form a new single column
def concat_feat(data):
    data['gender_location'] = data['gender'] + data['location_state']
    data['os_name_version'] = data['os_name'] + data['os_version'].astype(str)
    data['age_bucket'] = data.apply(lambda x: _age(x['age']), axis=1)
#         data['interactions'] = data['gender'] + data['age_bucket']
    
def check_nan(data=None, plot=True):
    '''
    Display missing values as a pandas dataframe.
    Parameters
    ----------
        data: DataFrame or named Series
        plot: bool, Default False
            Plots missing values in dataset as a heatmap
    
    Returns
    -------
        Matplotlib Figure:
            Heatmap plot of missing values
    '''

    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    df['missing_percent'] = missing_percent
    nan_values = df.set_index('features')['missing_percent']
    
    print_devider('Count and Percentage of missing value')
    if plot:
        plot_nan(nan_values)
    else:
        display(df)


def handle_nan(data,strategy='mean',fillna='mode',drop_outliers=False):
    
    """
    
    Function handles NaN values in a dataset for both categorical
    and numerical variables
    Args:
        strategy: Method of filling numerical features
        fillna: Method of filling categorical features
    """
    
#     identify_columns(data)
    if drop_outliers: remove_outliers(data)
    num_attributes, cat_attributes = load_attributes(data)
    
    if strategy=='mean':
        for item in num_attributes:
            data.loc[:,item] = data[item].fillna(data[item].mean())
    if fillna == 'mode':
        for item in cat_attributes:
            data.loc[:,item] = data[item].fillna(data[item].mode())
    else:
        for item in data[cat_attributes]:
            if item == 'customer_class':
                data.loc[:,item] = data[item].fillna(data[item].mean())
            else:
                data.loc[:,item] = data[item].fillna(fillna)
    check_nan(data)
            
    return data

def drop_cols(data,columns):
    data = data.drop(columns,axis=1)
    return data
    
def map_column(data,column_name,dict):
    data.loc[:,column_name] = data[column_name].map(dict)
    
    
def preprocess(dataframe=None,train=True,validation_path=None):
    if train:
        dataframe = dataframe[~dataframe['customer_class'].isnull()]
#         data = data.dropna(thresh=data.shape[1]*0.8)
        map_target(dataframe,'event_type')
        dataframe = handle_nan(dataframe,fillna='missing',drop_outliers=True)
        dataframe = drop_cols(dataframe,columns=['msisdn.1'])
        dataframe.to_pickle(PROCESSED_TRAIN_PATH)
        print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}')
        
    else:
        data = raw.read_data(path=validation_path)
        map_target(data,'event_type')
        data = handle_nan(data,fillna='missing',drop_outliers=True)
        data = drop_cols(data,columns=['msisdn.1'])

        data.to_pickle(PROCESSED_TEST_PATH)

        print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_TEST_PATH}')
    