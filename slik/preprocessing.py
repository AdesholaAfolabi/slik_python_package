import pandas as pd
# pd.options.mode.chained_assignment = None
import re,os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from IPython.display import display
import yaml,pathlib
from numpy import percentile
import pprint

import matplotlib.pyplot as plt
from .loadfile import read_file
from .utils import store_attribute,print_devider
from .plot_funcs import plot_nan
# from utils.config import TRAIN_PATH_CLICK,TRAIN_PATH_SMS,TEST_PATH,PROCESSED_TRAIN_PATH,PROCESSED_TEST_PATH


def bin_age(dataframe=None, age_col=None, add_prefix=True):

    '''
    The age attribute is binned into 5 categories (baby/toddler, child, young adult, mid age and elderly).
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    age_col: the name of the age column in the dataset. A string is expected
        The column to perform the operation on.
    add_prefix: Bool. Default is set to True
        add prefix to the column name. 
    Returns
    -------
        Dataframe with binned age attribute
    '''
    
    if dataframe is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(age_col,str):
        errstr = f'The given type for age_col is {type(age_col).__name__}. Expected type is string'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    
    if add_prefix:
        prefix_name = f'transformed_{age_col}'
    else:
        prefix_name = age_col
    
    bin_labels = ['Toddler/Baby', 'Child', 'Young Adult', 'Mid-Age', 'Elderly']
    data[prefix_name] = pd.cut(data[age_col], bins = [0,2,17,30,45,99], labels = bin_labels)
    data[prefix_name] = data[prefix_name].astype(str)
    
    return data


# concatenating name and version to form a new single column
# def concat_feat(data):
#     data['gender_location'] = data['gender'] + data['location_state']
#     data['os_name_version'] = data['os_name'] + data['os_version'].astype(str)
#     data['age_bucket'] = data.apply(lambda x: _age(x['age']), axis=1)
# #         data['interactions'] = data['gender'] + data['age_bucket']
    
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


def create_schema_file(dataframe,target_column, column_id, file_name,save=True, verbose=True):
    """Writes a map from column name to column datatype to a YAML file for a
    given dataframe. The schema format is as keyword arguments for the pandas
    `read_csv` function.
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    target_column: the name of the age column in the dataset. A string is expected
        The column to perform the operation on.
    column_id: Bool. Default is set to True
        add prefix to the column name.
    file_name:  str.
        name of the schema file you want to create.
    Returns
    -------
        Dataframe with binned age attribute
    """
    
    df = dataframe.copy()
    # ensure file exists
    output_path = f'./data/{file_name}'
    output_path = pathlib.Path(output_path)
    output_path.touch(exist_ok=True)

    # get dtypes schema
    datatype_map = {}
    datetime_fields = []
    for name, dtype in df.dtypes.iteritems():
        if 'datetime64' in dtype.name:
            datatype_map[name] = 'object'
            datetime_fields.append(name)
        else:
            datatype_map[name] = dtype.name
        
    print_devider('Creating Schema file')
    print('\nDone!. Schema file stored in data/')
    
    schema = dict(dtype=datatype_map, parse_dates=datetime_fields,
                  index_col=column_id, target_col = target_column)
    
    if verbose:
        display(schema)
    # write to YAML file
    if save:
        with output_path.open('w') as yaml_file:
            yaml.dump(schema, yaml_file)

        
def check_datefield(data, col):
    try:
        pd.to_datetime(data[col], infer_datetime_format=True)
        return True
    except:
        return False
    
    
def detect_fix_outliers(dataframe=None,target_column=None,n=1,num_features=None,fix_method='mean',verbose=True):
        
    '''
    Detect outliers present in the numerical features and fix the outliers present.
    Parameters:
    ------------------------
    data: DataFrame or name Series.
        Data set to perform operation on.
    num_features: List, Series, Array.
        Numerical features to perform operation on. If not provided, we automatically infer from the dataset.
    target_column: string
        The target attribute name. Not required for fixing, so it needs to be excluded.
    fix_method: mean or log_transformation.
        One of the two methods that you deem fit to fix the outlier values present in the dataset.

    Returns:
    -------
        Dataframe
            A new dataframe after removing outliers.
    
    '''

    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'") 

    data = dataframe.copy()
    
    df = data.copy()
    
    outlier_indices = []
    
    if num_features is None:
        if not isinstance(target_column,str):
            errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
            raise TypeError(errstr) 
        num_attributes, cat_attributes = get_attributes(data,target_column)
    else:
        num_attributes = num_features

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
        
        #apply any of the fix methods below to handle the outlier values
        if fix_method == 'mean':
            df.loc[:,column] = df[column].apply(lambda x : mean 
                                                        if x < lower or x > upper else x)
        elif fix_method == 'log_transformation':
            df.loc[:,column] = df[column].map(lambda i: np.log(i) if i > 0 else 0)
        else:
            raise ValueError("fix: must specify a fix method, one of [mean or log_transformation]")

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    if verbose:
        print_devider(f'Table idenifying more than {n} Outliers present in each record')
        display(data.loc[multiple_outliers])

    return df

def drop_uninformative_fields(dataframe):
    """After heavy cleaning, some of the fields left in the dataset track
    information that was never recorded for any of the loans in the dataset.
    These fields have only a single unique value or are all NaN, meaning
    that they are entirely uninformative. We drop these fields."""
    data = dataframe.copy()
    is_single = data.apply(lambda s: s.nunique()).le(1)
    single = data.columns[is_single].tolist()
    print_devider('Dropping uninformative fields')
    print(f'uninformative fields dropped:\n{single}')
    data = manage_columns(data,single,drop_columns=True)
    return data
    

def duplicate(data,columns,drop_duplicates=None):
    if drop_duplicates == 'rows':
        data = data.drop_duplicates()
        
    elif drop_duplicates == 'columns':
        data = data.drop_duplicates(subset=columns)
    
    elif drop_duplicates ==  None:
        pass

    else:
        raise ValueError("method: must specify a drop_duplicate method, one of ['rows' or 'columns']'")
    return data


def manage_columns(dataframe=None,columns=None, select_columns=False, drop_columns=False, drop_duplicates=None):
    
    '''
    Drop features from a pandas dataframe.
    Parameters
    ----------
        data: DataFrame or named Series
        columns: list of features you want to drop
        select_columns: Boolean True or False, default is False
            The columns you want to select from your dataframe. Requires a list to be passed into the columns param
        drop_columns: Boolean True or False, default is False
            The columns you want to drop from your dataset. Requires a list to be passed into the columns param
        drop_duplicates: 'rows' or 'columns', default is None
            Drop duplicate values across rows, columns. If columns, a list is required to be passed into the columns param
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe after dropping/selecting/removing duplicate columns or the original dataframe if params are left as default
    '''
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    if not isinstance(select_columns,bool):
        errstr = f'The given type for items is {type(select_columns).__name__}. Expected type is boolean True/False'
        raise TypeError(errstr)
        
    if not isinstance(drop_columns,bool):
        errstr = f'The given type for items is {type(drop_columns).__name__}. Expected type is boolean True/False'
        raise TypeError(errstr)

    if columns is None:
        raise ValueError("columns: A list/string is expected as part of the inputs to drop columns, got 'None'") 

    if select_columns and drop_columns:
        raise ValueError("Select one of select_columns or drop_columns at a time")  

      
    data = dataframe.copy()
    
    if select_columns:
        data = data[columns]
    
    if drop_columns:
        data = data.drop(columns,axis=1)
        
    data = duplicate(data,columns,drop_duplicates)
        
    return data

def featurize_datetime(dataframe=None, column_name=None, drop=True):
    '''
    Featurize datetime in the dataset to create new fields 
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    column_name: the name of the datetime column in the dataset. A string is expected
        The column to perform the operation on.
    drop: Bool. Default is set to True
        drop original datetime column. 
    Returns
    -------
        Dataframe with new datetime fields
            
    '''
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    if not isinstance(column_name,str):
        errstr = f'The given type is {type(column_name).__name__}. Specify target column name'
        raise TypeError(errstr)
        
    df = dataframe.copy()
    
    fld = df[column_name]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df.loc[:,column_name] = fld = pd.to_datetime(fld, infer_datetime_format=True,utc=True).dt.tz_localize(None)
    targ_pre_ = re.sub('[Dd]ate$', '', column_name)
    for n in ('Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df.loc[:,targ_pre_+n] = getattr(fld.dt,n.lower())
    for n in ['Week']:
        df.loc[:,targ_pre_+n] = getattr(fld.dt.isocalendar(),n.lower())
    df.loc[:,targ_pre_+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(column_name, axis=1, inplace=True)
    return df


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
        
#     if not isinstance(target_column,str):
#         errstr = f'The given type is {type(target_column).__name__}. Specify target column name'
#         raise TypeError(errstr)
        
    num_attributes = data.select_dtypes(exclude=['object', 'datetime64']).columns.tolist()
    cat_attributes = data.select_dtypes(include=['object']).columns.tolist()
    
    if target_column in num_attributes:
        num_attributes.remove(target_column)
    elif target_column in cat_attributes:
        cat_attributes.remove(target_column)
    else:
        pass
    return num_attributes, cat_attributes


def identify_columns(dataframe=None,target_column=None,id_column=None, high_dim=100, verbose=True, save_output=True):
    
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
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    if not isinstance(id_column,str):
        errstr = f'The given type for id_column is {type(id_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    num_attributes, cat_attributes = get_attributes(data,target_column)
        
    low_cat = []
    hash_features = []
#     dict_file = {}
    input_columns = [cols for cols in data.columns]
    input_columns.remove(target_column)
    input_columns.remove(id_column)
    if id_column in num_attributes:
        num_attributes.remove(id_column)
    else:
        cat_attributes.remove(id_column)
    
    print_devider('Identifying columns present in the data')
    print(f'Target column is {target_column}. Attribute in target column incldes:\n{list(data[target_column].unique())}\n')
    for item in cat_attributes:
        if data[item].nunique() > high_dim:
            hash_features.append(item)
        else:
            low_cat.append(item)
            
    print(f'Features with high cardinality:{hash_features}\n')        
    dict_file = dict(num_feat=num_attributes, cat_feat=cat_attributes,
                  hash_feat= hash_features, lower_cat = low_cat,
                 input_columns = input_columns, target_column = target_column,
                 id_column = id_column)
    
    if verbose:
        pprint.pprint(dict_file)
            
    if save_output:
        store_attribute(dict_file)

        print_devider('Saving Attributes in Yaml file')
        print('\nDone!. Data columns successfully identified and attributes are stored in data/\n')
        
    
def handle_cat_feat(data,fillna,cat_attr):
    if fillna == 'mode':
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(data[item].value_counts().index[0])
           
    else:
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(fillna)
    return data


def handle_nan(dataframe=None,target_name=None, strategy='mean',fillna='mode',\
               drop_outliers=True,thresh_y=75,thresh_x=75, verbose = True,
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
            
    Returns
    -------
        Pandas Dataframe:
            A new dataframe without the dropped features
    """
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    check_nan(data,verbose=verbose)
    df = check_nan.df
    
    if thresh_x:
        thresh = thresh_x/100
        initial_row = data.shape
        drop_row = data.shape[1] * thresh
        print(f'\nDropping rows with {thresh_x}% missing value')
        data = data.dropna(thresh=drop_row)
        
        
    if thresh_y:
        drop_col = df[df['missing_percent'] > thresh_y].features.to_list()
        print(f'\nDropping Columns with {thresh_y}% missing value: {drop_col}')
        data = manage_columns(data,columns = drop_col, drop_columns=True)
        print(f'\nNew data shape is {data.shape}')
    
    if drop_outliers:
        if target_name is None:
            raise ValueError("target_name: Expecting a str for the target_name, got 'None'")
        data = detect_fix_outliers(data,target_name,verbose=verbose,**kwargs)
        
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
    
def map_column(dataframe=None,column_name=None,items=None,add_prefix=True):
    
    '''
    Map values in  a pandas dataframe column with a dict.
    Parameters
    ----------
        data: DataFrame or named Series
        column_name: Name of pandas dataframe column to be mapped
        items: A dict with key and value to be mapped 
        add_prefix: Include a prefix of the target column in the dataset
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe without the dropped features
    '''
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(column_name,str):
        errstr = f'The given type for column_name is {type(column_name).__name__}. Expected type is str'
        raise TypeError(errstr)
  
    if not isinstance(items,dict):
        errstr = f'The given type for items is {type(items).__name__}. Expected type is dict'
        raise TypeError(errstr)
     
    data = dataframe.copy()
    
    print_devider('Mapping passed column')
    for key,value in items.items():
        print(f'{key} was mapped to {value}\n')
    
    if add_prefix:
        prefix_name = f'transformed_{column_name}'
    else:
        prefix_name = column_name

    data.loc[:,prefix_name] = data[column_name].map(items)
    return data


def map_target(dataframe=None,target_column=None,add_prefix=True,drop=False):
    
    '''
    Map target column in  a pandas dataframe column with a dict.
    Parameters
    ----------
        data: DataFrame or named Series
        column_name: Name of pandas dataframe column to be mapped
        add_prefix: Include a prefix of the target column in the dataset
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe without the dropped features
    '''
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for column_name is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    num_unique = data[target_column].unique()
    elem = data[target_column].value_counts()#.index.tolist()
    idx = elem.index.tolist()
    if len(num_unique) == 2:
        a = data[target_column].value_counts()[0]
        b = data[target_column].value_counts()[1]
        items = {}
        if a>b:
            items[idx[0]] = 0
            items[idx[1]] = 1
        else:
            items[idx[0]] = 1
            items[idx[1]] = 0
            
    elif len(num_unique) > 2:
        counter =[]
        for i in range(0,len(num_unique)):
            counter.append(i)
        items = dict(zip(num_unique,counter))
    
    else:
        raise ValueError("dataframe: The target column has only 1 unique value")
    
    print_devider('Mapping target columns')
    for key,value in items.items():
        print(f'{key} was mapped to {value}\n')
    if add_prefix:
        prefix_name = f'transformed_{target_column}'
    else:
        prefix_name = target_column
    data.loc[:,prefix_name] = data[target_column].map(items)
    
    if drop:
        data = manage_columns(data,target_column,drop_columns=True)
    return data
    
    
def preprocess_non_target_col(data=None,PROCESSED_DATA_PATH=None,verbose=True,
                              select_columns=None,**kwargs):
    
    if data is not None:
        if isinstance(data, pd.DataFrame):
            test_df = data
            if select_columns:
                test_df = manage_columns(train_df,columns=select_columns,select_columns=True)
        else:
            test_df = read_file(data, input_col= select_columns,**kwargs)

    num_attributes = test_df.select_dtypes(exclude=['object', 'datetime64']).columns.tolist()
    
    #how to indicate columns with outliers 
    data = detect_fix_outliers(dataframe=test_df,num_features=num_attributes,n=3,verbose=verbose)
    data = handle_nan(dataframe=data,n=3,drop_outliers=False,verbose=verbose,thresh_x=1,thresh_y=99,**kwargs)

    for column in data.columns:
        if 'age' in column.lower():
            match = re.search(r'(.*?)[Aa]ge.*', column).group()
            age_column = str(match)
            data = bin_age(data,age_column)

    for name, dtype in data.dtypes.iteritems():
        if 'datetime64' in dtype.name:
            print_devider('Featurize Datetime columns')
            print(f'column with datetime type:\n[{name}]\n') 
            data = featurize_datetime(data,name,False)#generic #methods to bin

        elif 'object' in dtype.name:
            output = check_datefield(data, name)
            if output:
                print_devider('Featurize Datetime columns')
                print(f'Inferred column with datetime type:\n[{name}]\n') 
                data = featurize_datetime(data,name,False)
        else:
            pass

    for column in data.columns:
        if 'gender' in column or 'sex' in column.lower():
            num_unique = data[column].unique()
            counter =[]
            for i in range(0,len(num_unique)):
                counter.append(i)
            items = dict(zip(num_unique,counter))
            data = map_column(data,column_name=column,items=items)

    drop_uninformative_fields(data)
    
    if verbose:
        print_devider('Display Top Five rows of the preprocessed data')
        display(data.head(5))    
    data.to_pickle(PROCESSED_DATA_PATH)
    print_devider('Preprocessed data saved')
    print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_DATA_PATH}')
    


def preprocess(data=None,target_column=None,train=False,select_columns=None,\
               verbose=True,processed_data_path=None,task='classification',**kwargs):
    # 1 - infer datetime
    #2 - pass select columns
    #3 check passed column is in dataframe column (throw error) 
    '''
    Map target column in  a pandas dataframe column with a dict.
    Parameters
    ----------
        data: DataFrame or named Series
        target_column: Name of pandas dataframe column to be mapped
        select_columns:
        processed_data_path:
        verbose:
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe without the dropped features
    '''
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series or a data path, got None")
    
#     if train == False and attribute_path == None:
#         raise ValueError("dataframe: Expecting attribue path, got None")
        
    if processed_data_path is None:
        raise ValueError("processed_data_path: Expecting a path to store the preprocessed data")
        
    if os.path.exists(processed_data_path):
        pass
    else:
        os.mkdir(processed_data_path)
        
    print_devider(f'The task for preprocessing is {task}')    
    
    if task == 'classification':
        if train:
            if not isinstance(target_column,str):
                errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
                raise TypeError(errstr) 

            PROCESSED_TRAIN_PATH = os.path.join(processed_data_path, 'train_data.pkl')
            
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    train_df = data
                    if select_columns:
                        train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
                else:
                    train_df = read_file(data, input_col= select_columns,**kwargs)
                    
            data = handle_nan(dataframe=train_df,target_name=target_column,verbose=verbose,n=3)# change handlena to fillna #default threshold judgement
            data = map_target(data,target_column=target_column,drop=True)
            prefix_name = f'transformed_{target_column}'
            for column in data.columns:
                if 'age' in column.lower():
                    print_devider('Bucketize Age columns')
                    print(f'Inferred age column: [{column}]')
                    match = re.search(r'(.*?)[Aa]ge.*', column).group()
                    age_column = str(match)
                    data = bin_age(data,age_column)

            for name, dtype in data.dtypes.iteritems():
                if 'datetime64' in dtype.name:
                    print_devider('Featurize Datetime columns')
                    print(f'column with datetime type:\n[{name}]\n') 
                    data = featurize_datetime(data,name,False)#generic #methods to bin
     
                elif 'object' in dtype.name:
                    output = check_datefield(data, name)
                    if output:
                        print_devider('Featurize Datetime columns')
                        print(f'Inferred column with datetime type:\n[{name}]\n') 
                        data = featurize_datetime(data,name,False)
                else:
                    pass

            for column in data.columns:
                if 'gender' in column or 'sex' in column.lower():
                    num_unique = data[column].unique()
                    counter =[]
                    for i in range(0,len(num_unique)):
                        counter.append(i)
                    items = dict(zip(num_unique,counter))
                    data = map_column(data,column_name=column,items=items)

            drop_uninformative_fields(data)

            create_schema_file(data,target_column=prefix_name,file_name='schema.yaml',\
                               verbose=verbose,column_id=data.columns[0])
            if verbose:
                print_devider('\nDisplay Top Five rows of the preprocessed data')
                display(data.head(5))    

            data.to_pickle(PROCESSED_TRAIN_PATH)
            print_devider('Preprocessed data saved')
            print(f'\nDone!. Input data preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}\n')
        
        else:
            PROCESSED_TEST_PATH = os.path.join(processed_data_path, 'validation_data.pkl')
            preprocess_non_target_col(data=data,PROCESSED_DATA_PATH = PROCESSED_TEST_PATH,verbose=verbose,\
                                     select_columns=select_columns,**kwargs)
    
    elif task == 'clustering':
        PROCESSED_CLUSTER_PATH = os.path.join(processed_data_path, 'preprocessed_cluster_data.pkl')
        preprocess_non_target_col(data =data, PROCESSED_DATA_PATH = PROCESSED_CLUSTER_PATH,verbose=verbose,\
                                 select_columns=select_columns,**kwargs)
        
    else:
        raise ValueError("task: Expecting one of ['classification','clustering']")
        