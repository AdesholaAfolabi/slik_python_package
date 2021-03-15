import pandas as pd
# pd.options.mode.chained_assignment = None


import re,os
import pathlib


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from IPython.display import display
import yaml,pathlib
from numpy import percentile
import pprint
import matplotlib.pyplot as plt
from .loadfile import read_file
from .utils import store_attribute,print_devider,HiddenPrints
from .plot_funcs import plot_nan

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
        errstr = f'The given type for age_col is {type(age_col).__name__}. Expected type is a string'
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




def create_schema_file(dataframe,target_column,id_column,file_name,save=True, verbose=True):
    """
    
    Writes a map from column name to column datatype to a YAML file for a
    given dataframe.

    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.

    target_column: the name of the target column in the dataset. A string is expected
        The column to perform the operation on.
        
    id_column: str
        Unique Identifier column.
        
    file_name:  str.
        name of the schema file you want to create.
        
    save: Bool. Default is set to True
        save schema file to file path.
        
    verbose: Bool. Default is set to True
        display dataframe print statements.

    Output
    -------
        A schema file is created in the data directory
    """
    
    df = dataframe.copy()
    # ensure file exists
    
    output_path = f'{file_name}/metadata'
    output_path = pathlib.Path(output_path)
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)
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
    print('Schema file stored in output_path')
    
    schema = dict(dtype=datatype_map, parse_dates=datetime_fields,
                  index_col=id_column, target_col = target_column)
    
    if verbose:
        display(schema)
    # write to YAML file
    if save:
        with open(f'{output_path}/schema.yaml', 'w') as yaml_file:
            yaml.dump(schema, yaml_file)

        
def check_datefield(dataframe=None, column=None):
    '''
    Check if a column is a datefield and Returns a Bool.
    
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    column: str
        The column to perform the operation on.
          
    '''
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series") 
        
    try:
        pd.to_datetime(dataframe[column], infer_datetime_format=True)
        return True
    except:
        return False
    
    
def detect_fix_outliers(dataframe=None,target_column=None,n=1,num_features=None,fix_method='mean',verbose=True):
        
    '''
    Detect outliers present in the numerical features and fix the outliers present.
    Parameters:
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    num_features: List, Series, Array.
        Numerical features to perform operation on. If not provided, we automatically infer from the dataset.
        
    target_column: string
        The target attribute name. Not required for fixing, so it needs to be excluded.
        
    fix_method: Method of fixing outliers present in the data. mean or log_transformation. Default is 'mean'

    n: integar
        A value to determine whether there are multiple outliers in a record, which is highly dependent on the
        number of features that are being checked. 

    fix_method: mean or log_transformation.

        One of the two methods that you deem fit to fix the outlier values present in the dataset.

    Returns:
    -------
    Dataframe:
        dataframe after removing outliers.
    
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

    """

    Dropping fields that have only a single unique value or are all NaN, meaning
    that they are entirely uninformative.

    """
    data = dataframe.copy()
    is_single = data.apply(lambda s: s.nunique()).le(1)
    single = data.columns[is_single].tolist()
    print_devider('Dropping uninformative fields')
    print(f'uninformative fields dropped: {single}')
    data = manage_columns(data,single,drop_columns=True)
    return data
    

def drop_duplicate(dataframe=None,columns=None,method=None):
    '''
    Drop duplicate values across rows, columns in the dataframe.
    Parameters:
    ------------------------
    
    
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    columns: List/String.
        list of column names 
    drop_duplicates: 'rows' or 'columns', default is None
    
        Drop duplicate values across rows, columns. If columns, a list is required to be passed into the columns param
    Returns:
    -------
    Dataframe:
        dataframe after dropping duplicates.
    '''
    
    if method == 'rows':
        dataframe = dataframe.drop_duplicates()
        
    elif method == 'columns':
        if columns is None:
            raise ValueError("columns: A list/string is expected as part of the inputs to columns, got 'None'")
        dataframe = dataframe.drop_duplicates(subset=columns)
    
    elif method ==  None:
        pass

    else:
        raise ValueError("method: must specify a drop_duplicate method, one of ['rows' or 'columns']'")
    return dataframe


def manage_columns(dataframe=None,columns=None, select_columns=False, drop_columns=False, drop_duplicates=None):
    
    '''
    Manage operations on pandas dataframe based on columns. Operations include selecting 
    columns, dropping column and dropping duplicates.
    
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
            A new dataframe after dropping/selecting/removing duplicate columns or the original 
            dataframe if params are left as default
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
        
    data = drop_duplicate(data,columns,method=drop_duplicates)
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
    for n in ('Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear','Week',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df.loc[:,targ_pre_+n] = getattr(fld.dt,n.lower())
    df.loc[:,targ_pre_+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(column_name, axis=1, inplace=True)
    return df


def get_attributes(data=None,target_column=None):
    
    '''
    Returns the categorical features and Numerical features in a pandas dataframe
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


def identify_columns(dataframe=None,target_column=None,id_column=None, high_dim=100, verbose=True, output_path=None):
    
    """
        Identifies numerical attributes ,categorical attributes with sparse features 
        and categorical attributes with lower features present in the data and saves in a yaml file.
        
     Parameters:
    -----------
        dataframe: DataFrame or named Series 
        
        target_column: str
            Label or Target column.
            
        id_column: str
            unique identifier column.
            
        high_dim: int, default 100
            Integer to identify categorical attributes greater than 100 features
            
        verbose: Bool, default=True
            display print statement
            
        output_path: str
            path to where the yaml file is saved.   
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    if not isinstance(id_column,str):
        errstr = f'The given type for id_column is {type(id_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    output_path = f'{output_path}metadata'
    output_path = pathlib.Path(output_path)
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)
    output_path.touch(exist_ok=True)

        
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
    
    store_attribute(dict_file,output_path)

    print_devider('Saving Attributes in Yaml file')
    print(f'\nData columns successfully identified and attributes are stored in {output_path}\n')
        
    
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
    Handle missing values present in pandas dataframe. Outliers are treated before handling \
    missing values.
    Args:
    ------------------------
    data: DataFrame or name Series.
        Data set to perform operation on.
        
    target_name: str
        Name of the target column
        
    strategy: str. Default is 'mean'
        Method of filling numerical features
        
    fillna: str. Default is 'mode'
        Method of filling categorical features
        
    drop_outliers: bool, Default True
        Drops outliers present in the data.
        
    thresh_x: Int, Default is 75.
        Threshold for dropping rows with missing values 
        
    thresh_y: In, Default is 75.
        Threshold for dropping columns with missing value
        
    verbose: Bool. default is True.
        display pandas dataframe print statements
            
    Returns
    -------
        Pandas Dataframe:
           Dataframe without missing values
    """
    
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    check_nan(data,verbose=verbose)
    df = check_nan.df
    
    if thresh_x:
        thresh = thresh_x/100
        initial_row = data.shape[0]
        drop_row = data.shape[1] * thresh
        print(f'\nDropping rows with {thresh_x}% missing value')
        data = data.dropna(thresh=drop_row)
        print(f'\n       Number of records dropped from {initial_row} to {data.shape[0]}')
        
        
    if thresh_y:
        drop_col = df[df['missing_percent'] > thresh_y].features.to_list()
        print(f'\nDropping Columns with {thresh_y}% missing value: {drop_col}')
        data = manage_columns(data,columns = drop_col, drop_columns=True)
        print(f'\nNew data shape is {data.shape}')
    
    if drop_outliers:
        if target_name is None:
            raise ValueError("target_name: Expecting Target_column ")
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
        
        column_name: str. 
            Name of pandas dataframe column to be mapped
            
        items: Dict, default is None
            A dict with key and value to be mapped 
            
        add_prefix: Bool, default is True
            Include a prefix of the target column in the dataset
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe with mapped features.
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
        
        column_name: str
            Name of pandas dataframe column to be mapped
            
        add_prefix: Bool. Default is True
            Include a prefix of the target column in the dataset
            
        drop: Bool. Default is True
            drop original target column name
    
    Returns
    -------
        Pandas Dataframe:
            A new dataframe with mapped target column
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
            
    test_df = drop_uninformative_fields(test_df)
    num_attributes = test_df.select_dtypes(exclude=['object', 'datetime64']).columns.tolist()
    
    with HiddenPrints(): 
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
                print(f'column with datetime type: [{name}]\n') 
                data = featurize_datetime(data,name,False)#generic #methods to bin

            elif 'object' in dtype.name:
                output = check_datefield(data, name)
                if output:
                    print_devider('Featurize Datetime columns')
                    print(f'Inferred column with datetime type: [{name}]\n') 
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
    
    if verbose:
        print_devider('Display Top Five rows of the preprocessed data')
        display(data.head(5))
    
    data.to_pickle(PROCESSED_DATA_PATH)
    print_devider('Preprocessed data saved')
    print(f'\nDone!. Input data has been preprocessed successfully and stored in {PROCESSED_DATA_PATH}')
    


def preprocess(data=None,target_column=None,train=False,select_columns=None,\
               verbose=True,project_path=None,**kwargs): 
    
    '''
    Automatically preprocess dataframe/file-path. Handles missing value, Outlier treatment,
    feature engineering. 
    
    Parameters
    ----------
        data: DataFrame or named Series
            Dataframe or dath path to the data
            
        target_column: String
            Name of pandas dataframe target column
            
        train: Bool, default is True
            
        select_columns: List
            List of columns to be used
            
        project_path: Str
            Path to where the preprocessed data will be stored
            
        verbose:Bool. Default is  True
    
    Returns
    -------
        Pandas Dataframe:
            Returns a clean dataframe in the filepath
    '''
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series or a data path, got None")
        
    if project_path is None:
        raise ValueError("project_path: Expecting a path to store the preprocessed data")
        
        
    project_path = os.path.join(project_path, 'data')
    if os.path.exists(project_path):
        pass
    else:
        os.mkdir(project_path)
    
    if data is not None:
        if isinstance(data, pd.DataFrame):
            train_df = data
            if select_columns:
                train_df = manage_columns(train_df,columns=select_columns,select_columns=True)
        else:
            train_df = read_file(data, input_col= select_columns,**kwargs)
        
    if target_column:
        
        row_length = train_df.shape[0]
        one_percent = 1/100 * row_length
        if train_df[target_column].nunique() > one_percent:
            task = 'regression'
        if train_df[target_column].nunique() < one_percent:
            task = 'classification'

    if train:        
        if task == 'classification':
            if not isinstance(target_column,str):
                errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
                raise TypeError(errstr)
            print(f'\nThe task for preprocessing is {task}')

            PROCESSED_TRAIN_PATH = os.path.join(project_path, 'train_data.pkl')
            data = handle_nan(dataframe=train_df,target_name=target_column,verbose=verbose,n=3)
            data = map_target(data,target_column=target_column,drop=True)
            prefix_name = f'transformed_{target_column}'
            for column in data.columns:
                if 'age' in column.lower():
                    print_devider('Bucketize Age columns')
                    print(f' Inferred age column: [{column}]')
                    match = re.search(r'(.*?)[Aa]ge.*', column).group()
                    age_column = str(match)
                    data = bin_age(data,age_column)

            for name, dtype in data.dtypes.iteritems():
                if 'datetime64' in dtype.name:
                    print_devider('Featurize Datetime columns')
                    print(f'column with datetime type: [{name}]\n') 
                    data = featurize_datetime(data,name,False)

                elif 'object' in dtype.name:
                    output = check_datefield(data, name)
                    if output:
                        print_devider('Featurize Datetime columns')
                        print(f'Inferred column with datetime type: [{name}]\n') 
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
                    data = manage_columns(data,columns=column,drop_columns=True)

            data = drop_uninformative_fields(data)

            create_schema_file(data,target_column=prefix_name,file_name=project_path,\
                               verbose=verbose,id_column=data.columns[0])
            if verbose:
                print_devider('\nDisplay Top Five rows of the preprocessed data')
                display(data.head(5))    

            data.to_pickle(PROCESSED_TRAIN_PATH)
            print_devider('Preprocessed data saved')
            print(f'\n Input data preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}\n')
            
        elif task == 'clustering':
            PROCESSED_CLUSTER_PATH = os.path.join(project_path, 'preprocessed_cluster_data.pkl')
            preprocess_non_target_col(data = train_df, PROCESSED_DATA_PATH = PROCESSED_CLUSTER_PATH,verbose=verbose,\
                                 select_columns=select_columns,**kwargs)
        else:
            raise ValueError("task: Does not support NLP tasks")

    else:
        PROCESSED_TEST_PATH = os.path.join(project_path, 'validation_data.pkl')
        preprocess_non_target_col(data=train_df,PROCESSED_DATA_PATH = PROCESSED_TEST_PATH,verbose=verbose,\
                                 select_columns=select_columns,**kwargs)

    