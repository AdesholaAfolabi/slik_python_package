from matplotlib.pyplot import plot
import pandas as pd
# pd.options.mode.chained_assignment = None
import re, os
import pathlib
import numpy as np
from collections import Counter
from IPython.display import display
import yaml, pathlib
from difflib import get_close_matches
import pprint
from .loadfile import read_file
from .utils import store_attribute, print_divider, HiddenPrints
from pandas.errors import ParserError
from .plot_funcs import plot_nan


def bin_age(dataframe=None, age_col=None, add_prefix=True):

    """
    The age attribute in a DataFrame is binned into 5 categories:
    (baby/toddler, child, young adult, mid age and elderly).

    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    age_col: str.
        The column to perform the operation on.
        
    add_prefix: Bool. Default is set to True
        add prefix to the column name. 

    Returns
    -------
    Dataframe with binned age attribute
    """
    if dataframe is None:
        raise ValueError("dataframe: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(age_col, str):
        errstr = f'The given type for age_col is {type(age_col).__name__}. Expected type is a string'
        raise TypeError(errstr)
        
    data = dataframe.copy()
    
    if add_prefix:
        prefix_name = f'binned_{age_col}'
    else:
        prefix_name = age_col
    
    bin_labels = ['Toddler/Baby', 'Child', 'Young Adult', 'Mid-Age', 'Elderly']
    data[prefix_name] = pd.cut(data[age_col], bins = [0,2,17,30,45,99], labels = bin_labels)
    data[prefix_name] = data[prefix_name].astype(str)
    return data


def change_case(dataframe, columns=None, case='lower', inplace=False):
    """
    Change the case of a pandas series to either upper or lower case

    Parameters
    ----------
    dataframe: Dataframe or named Series

    columns: str, list
        The column or list of columns to perform the operation on

    case: str. Default is set to lower
        Indicates the type of operation to perform

    inplace: bool. Default is set to False
        Indicates if changes should by made within the dataframe or not.

    Returns
    -------
    Pandas Dataframe:
    """

    if type(dataframe) != pd.DataFrame and type(dataframe) != pd.Series:
        raise ValueError(
            "data: Expecting a DataFrame or Series, got None"
        )

    if isinstance(columns, str):
        columns = [columns]

    if not isinstance(columns, list):
        raise TypeError(
            f'The given type for column is {type(columns).__name__}. Expected type is a string or list'
        )

    if not inplace:
        dataframe = dataframe.copy()

    def capitalize_case(x):
        """
        Capitalize Case Function
        
        Parameters
        ----------
        x: str
            The string to be capitalized
        
        Returns
        -------
        A capitalized string
        """

        return ' '.join([
            item.capitalize()
            for item in x.split(' ')
        ])

    def perform_case(func):
        dataframe[columns] = dataframe[columns].applymap(func)

    action = {
        'lower': lambda x: x.lower(),
        'upper': lambda x: x.upper(),
        'capitalize': lambda x: capitalize_case(x)
    }

    if case in action.keys():
        perform_case(action[case])

        if not inplace:
            return dataframe
    else:
        raise ValueError(f"case: expected one of upper,lower or captitalize got {case}")
    

def check_nan(dataframe=None, plot=False, display_inline=True):
    
    """
    Display missing values as a pandas dataframe and give a proportion
    in terms of percentages.

    Parameters
    ----------
    data: pandas DataFrame or named Series
    
    plot: bool, Default False
        Plots missing values in dataset as a heatmap
        
    display_inline: bool, Default False
        shows missing values in the dataset as a dataframe
    Returns
    -------
    Matplotlib Figure:
        Bar plot of missing values
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    df = data.isna().sum()
    df = df.reset_index()
    df.columns = ['features', 'missing_counts']

    missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
    df['missing_percent'] = missing_percent
    nan_values = df.set_index('features')['missing_percent']
    
    
    if plot:
        plot_nan(nan_values)
    if display_inline:
        print_divider('Count and Percentage of missing value')
        display(df)
    check_nan.df = df


def create_schema_file(dataframe, target_column, id_column, project_path='.', save=True, display_inline=True):

    """
    
    A data schema of column names and types are automatically inferred and
    saved in a YAML file

    Parameters
    ------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.

    target_column: the name of the target column in the dataset. A string is expected
        The column to perform the operation on.
        
    id_column: str
        Unique Identifier column.
        
    project_path:  str.
        The path of the schema file you want to create.
        
    save: Bool. Default is set to True
        save schema file to file path.
        
    display_inline: Bool. Default is set to True
        display dataframe print statements.

    Returns
    -------
    file path:
        A schema file is created in the data directory
    """
    df = dataframe.copy()
    # ensure file exists

    try:
        os.mkdir(project_path)
    except:
        pass


    # get dtypes schema
    datatype_map = {}
    datetime_fields = []
    for name, dtype in df.dtypes.iteritems():
        if 'datetime64' in dtype.name:
            datatype_map[name] = dtype.name
            datetime_fields.append(name)
        else:
            datatype_map[name] = dtype.name
    
    
    schema = dict(dtype=datatype_map)
    
    # write to YAML file
    if save:
        output_path =  os.path.join(project_path,'metadata')
        output_path = pathlib.Path(output_path)
        if os.path.exists(output_path):
            pass
        else:
            os.mkdir(output_path)
        output_path.touch(exist_ok=True)
        with open(f'{output_path}/schema.yaml', 'w') as yaml_file:
            yaml.dump(schema, yaml_file)
            
    if display_inline:
        print_divider('Creating Schema file')
        display(schema)
        if save:
            print(f'\n\nSchema file stored in {output_path}')

        
def check_datefield(dataframe=None, column=None):

    """
    Check if a column is a datefield and Returns a Bool.
    
    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    column: str
        The column name to perform the operation on.
          
    Returns
    -------
    Boolean:
        Returns True if the data point is a datefield.
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series") 
        
    if not isinstance(column, str):
        errstr = f'The given type for column is {type(column).__name__}. Expected type is a string'
        raise TypeError(errstr)

    if isinstance(dataframe.dtypes[column],object):
        return False
    if pd.to_datetime(dataframe[column], infer_datetime_format=True):
        return True

    
def detect_fix_outliers(dataframe=None,target_column=None,n=1,num_features=None,fix_method='mean',display_inline=True):
        
    """
    Detect outliers present in the numerical features and fix the outliers 
    present.

    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    num_features: List, Series, Array.
        Numerical features to perform operation on. If not provided, 
        we automatically infer from the dataset.
        
    target_column: string
        The target attribute name.
        
    fix_method: mean or log_transformatio. Default is 'mean'
        Method of fixing outliers present in the data. mean or 
        log_transformation. 

    n: integer
        A value to determine whether there are multiple outliers in a record,
        which is highly dependent on the number of features that are being checked. 

    display_inline: Bool. Default is True.
        Display the outliers present in the data in form of a dataframe.

    Returns
    -------
    Dataframe:
        dataframe after removing outliers.
    
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'") 

    # data = dataframe.copy()
    
    df = dataframe.copy()
    
    outlier_indices = []
    
    if num_features is None:
        if not isinstance(target_column,str):
            errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
            raise TypeError(errstr) 
        num_attributes, cat_attributes = get_attributes(dataframe,target_column)
    else:
        num_attributes = num_features

    for column in num_attributes:
        
        dataframe.loc[:,column] = abs(dataframe[column])
        mean = dataframe[column].mean()

        #calculate the interquartlie range
        q25, q75 = np.percentile(dataframe[column].dropna(), 25), np.percentile(dataframe[column].dropna(), 75)
        iqr = q75 - q25

        #calculate the outlier cutoff
        cut_off = iqr * 1.5
        lower,upper = q25 - cut_off, q75 + cut_off

        #identify outliers
        # Determine a list of indices of outliers for feature col
        outlier_list_col = dataframe[(dataframe[column] < lower) | (dataframe[column] > upper)].index

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

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    
    if display_inline:
        print_divider(f'Table identifying at least {n} outliers in a row')
        display(dataframe.loc[multiple_outliers])

    return df


def drop_uninformative_fields(dataframe = None, exclude= None, display_inline=True):

    """
    Drop fields that have only a single unique value or are all NaN, meaning
    that they are entirely uninformative.

    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.

    exclude: string/list.
        A column or list of columns you want to exclude from being dropped.
        
    display_inline: Bool. Default is True.
        Display print statements.
        
    Returns
    -------
    Dataframe:
        dataframe after dropping uninformative fields.

    """

    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

    data = dataframe.copy()
    is_single = data.apply(lambda s: s.nunique()).le(1)
    single = data.columns[is_single].tolist()
    if exclude:
        single = [column_name for column_name in single if column_name not in exclude]
    if display_inline:
        print_divider('Dropping uninformative fields')
        print(f'uninformative fields dropped: {single}')
    data = manage_columns(data,single,drop_columns=True,drop_duplicates=None)
    return data
    

def drop_duplicate(dataframe=None,columns=None,method='rows',display_inline=True):

    """
    Drop duplicate values across rows, columns in the dataframe.

    Parameters
    -----------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
    columns: List/String.
        list of column names 
    method: 'rows' or 'columns', default is 'rows'
        Drop duplicate values across rows, columns. 
    
    display_inline: Bool. Default is True.
        Display print statements.
        
    Returns
    -------
    Dataframe:
        dataframe after dropping duplicates.
    """
    if method == 'rows':
        dataframe = dataframe.drop_duplicates()
        
    elif method == 'columns':
        if columns is None:
            raise ValueError("columns: A list/string is expected as part of the inputs to columns, got 'None'")
        dataframe = dataframe.drop_duplicates(subset=columns)
    
    elif method is None:
        pass
        
    else:
        raise ValueError("method: must specify a drop_duplicate method, one of ['rows' or 'columns']'")
        
    if display_inline:
        print_divider(f'Dropping duplicates across the {method}')
        print(f'New datashape is {dataframe.shape}')
    return dataframe


def featurize_datetime(dataframe=None, column_name=None, date_features=None, drop=True):
    """
    Featurize datetime in the dataset to create new fields such as 
    the Year, Month, Day, Day of the week, Day of the year, Week,
    end of the month, start of the month, end of the quarter, 
    start of a quarter, end of the year, start of the year

    Parameters
    ------------------------
    dataframe: DataFrame or name Series.
        Data set to perform operation on.
        
    column_name: String
        The column to perform the operation on.

    date_features: List. 
        A list of new datetime features to include in the dataset. 
        Expected list should contain either of the elements in this list\
                                ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',\
                                'Week','Is_month_end', 'Is_month_start', 'Is_quarter_end',\
                                'Hour','Minute','Is_quarter_start', 'Is_year_end', 'Is_year_start',\
                                'Date']
        
    drop: Bool. Default is set to True
        drop original datetime column. 

    Returns
    -------
    Dataframe:
        Dataframe with new datetime fields
            
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    if not isinstance(column_name,str):
        errstr = f'The given type is {type(column_name).__name__}. Specify target column name'
        raise TypeError(errstr)
        
    df = dataframe.copy()
    expected_list = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear','Week',\
            'Is_month_end', 'Is_month_start', 'Is_quarter_end','Hour','Minute',\
                 'Is_quarter_start', 'Is_year_end', 'Is_year_start','Date']
    if date_features is None:
        date_features = expected_list
        for elem in date_features:
            if elem not in expected_list:
                raise KeyError(f'List should contain any of {expected_list}')

    fld = df[column_name]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df.loc[:,column_name] = fld = pd.to_datetime(fld, infer_datetime_format=True,utc=True).dt.tz_localize(None)
    targ_pre_ = re.sub('[Dd]ate$', '', column_name)
    for n in date_features:
        df.loc[:,targ_pre_+n] = getattr(fld.dt,n.lower())
        if n.lower() == 'dayofweek':
            df.loc[:,targ_pre_+'Isweekend'] = df.loc[:,targ_pre_+n] > 4

#     df.loc[:,targ_pre_+'Elapsed'] = fld.astype(np.int64) // 10**9
    if drop: df.drop(column_name, axis=1, inplace=True)
    return df


def get_attributes(data=None,target_column=None):
    
    """
    Returns the categorical features and Numerical features(in a pandas dataframe) as a list 

    Parameters
    -----------
    data: DataFrame or named Series
        Data set to perform operation on.
        
    target_column: str
        Label or Target column

    Returns
    -------
        List
            A list of all the categorical features and numerical features in a dataset.
    """
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
        
    num_attributes = data.select_dtypes(include=np.number).columns.tolist()
    cat_attributes = [x for x in data.columns if x not in num_attributes]
    
    if target_column in num_attributes:
        num_attributes.remove(target_column)
    elif target_column in cat_attributes:
        cat_attributes.remove(target_column)
    else:
        pass
    return num_attributes, cat_attributes
        
    
def _handle_cat_feat(data,fillna,cat_attr):

    """
    Handle missing values present in a categorical pandas dataframe. 

    Parameters
    -----------
    data: DataFrame or name Series.
        Data set to perform operation on.
        
    fillna: str. Default is 'mode'
        Method of filling categorical features
        
    cat_attr: list.
        List of categorical attributes to pass.
            
    Returns
    -------
        Pandas Dataframe:
           Dataframe without missing values
    """
    if fillna == 'mode':
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(data[item].value_counts().index[0])
           
    else:
        for item in cat_attr:
            data.loc[:,item] = data[item].fillna(fillna)
    return data


def handle_nan(dataframe=None,target_name=None, strategy='mean',fillna='mode',\
               drop_outliers=True,thresh_y=75,thresh_x=75, display_inline = True,
               **kwargs):
    
    """
    
    Handle missing values present in a pandas dataframe. 
    
    Take care of missing values in the data both cateforical and 
    numerical features by dropping or filling missing values. Using the 
    threshold parameter you can also drop missing values present in the data.
    Outliers are treated before handling missing values by default.

    Parameters
    -----------
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
        Threshold for dropping rows with missing values.  
        
    thresh_y: In, Default is 75.
        Threshold for dropping columns with missing value
        
    display_inline: Bool. default is True.
        display pandas dataframe print statements
            
    Returns
    -------
        Pandas Dataframe:
           Dataframe without missing values
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
        
    data = dataframe.copy()
    check_nan(data,display_inline=False,plot=False)
    df = check_nan.df
    
    
    thresh = thresh_x/100
    initial_row = data.shape[0]
    drop_row = data.shape[1] * thresh
    data = data.dropna(thresh=drop_row)
    dropped_records = initial_row - data.shape[0]
    
    drop_col = df[df['missing_percent'] > thresh_y].features.to_list()
    data = manage_columns(data,columns = drop_col, drop_columns=True,drop_duplicates=None)
    
    if display_inline:
        print(f'\nDropping rows with {thresh_x}% missing value: Number of records dropped is {dropped_records}')
        print(f'\nDropping Columns with {thresh_y}% missing value: {drop_col}')
        print(f'\nNew data shape is {data.shape}')
        
    if drop_outliers:
        if target_name is None:
            raise ValueError("target_name: Expecting Target_column ")
        data = detect_fix_outliers(data,target_name,display_inline=False)
        
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
        
    data = _handle_cat_feat(data,fillna,cat_attributes)
    return data
    
def identify_columns(dataframe=None,target_column=None,id_column=None, high_dim=100, display_inline=True, project_path=None):
    
    """
    Identifies numerical attributes ,categorical attributes with sparse features 
    and categorical attributes with lower features present in the data and saves
    the output in a yaml file.
        
    Parameters
    -----------
    dataframe: DataFrame or named Series 
    
    target_column: str
        Label or Target column.
        
    id_column: str
        unique identifier column.
        
    high_dim: int, default 100
        Integer to identify categorical attributes greater than 100 observations
        
    display: Bool, default=True
        display print statement
        
    project_path: str
        path to where the yaml file is saved.   
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)
        
    if not isinstance(id_column,str):
        if type(id_column).__name__ == 'NoneType':
            errstr = f'Nothing was passed for param id_column'
        else:
            errstr = f'The given type for id_column is {type(id_column).__name__}. Expected type is str'
        raise TypeError(errstr)

    if not isinstance(project_path,str):
        if type(project_path).__name__ == 'NoneType':
            errstr = f'Nothing was passed for param project_path'
        else:
            errstr = f'The given type for id_column is {type(project_path).__name__}. Expected type is str'
        raise TypeError(errstr)

    try:
        os.mkdir(project_path)
    except:
        pass

    output_path =  os.path.join(project_path,'metadata')
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
    input_columns = [cols for cols in data.columns]
    input_columns.remove(target_column)
    input_columns.remove(id_column)
    if id_column in num_attributes:
        num_attributes.remove(id_column)
    else:
        cat_attributes.remove(id_column)
    
    for item in cat_attributes:
        if data[item].nunique() > high_dim:
            hash_features.append(item)
        else:
            low_cat.append(item)

    datetime_fields = []
    for name, dtype in data.dtypes.iteritems():
        if 'datetime64' in dtype.name:
            datetime_fields.append(name)
            
           
    dict_file = dict(num_feat=num_attributes, cat_feat=cat_attributes,
                  high_card_feat= hash_features, lower_cat = low_cat,
                 input_columns = input_columns, parse_dates=datetime_fields,
                  target_column = target_column,
                 id_column = id_column)
    
    if display_inline:
        print_divider('Identifying columns present in the data')
        print(f'Target column is {target_column}. Attribute in target column:{list(data[target_column].unique())}\n')
        
        print(f'Features with high cardinality:{hash_features}\n') 
        pprint.pprint(dict_file)
        
        print(f'\nAttributes are stored in {output_path}\n')

    store_attribute(dict_file,output_path)


def manage_columns(dataframe=None,columns=None, select_columns=False, drop_columns=False, drop_duplicates=None):
    
    """
    Manage operations on pandas dataframe based on columns. Operations include 
    selecting of columns, dropping column and dropping duplicates.
    
    Parameters
    ----------
    dataframe: DataFrame or named Series
    
    columns: used to specify columns to be selected, dropped or used in dropping duplicates. 
    
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
    """
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
        
    if drop_duplicates:
        data = drop_duplicate(data,columns,method=drop_duplicates)
        
    return data


def map_column(dataframe=None,column_name=None,items=None,add_prefix=True):
    
    """
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
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(column_name,str):
        errstr = f'The given type for column_name is {type(column_name).__name__}. Expected type is str'
        raise TypeError(errstr)
  
    if not isinstance(items,dict):
        errstr = f'The given type for items is {type(items).__name__}. Expected type is dict'
        raise TypeError(errstr)
     
    data = dataframe.copy()
    
    print_divider('Mapping passed column')
    for key,value in items.items():
        print(f'{key} was mapped to {value}\n')
    
    if add_prefix:
        prefix_name = f'transformed_{column_name}'
    else:
        prefix_name = column_name

    data.loc[:,prefix_name] = data[column_name].map(items)
    return data


def map_target(dataframe=None,target_column=None,add_prefix=True,drop=False,display_inline=True):

    """
    Map target column in  a pandas dataframe column with a dict.
    This can be applied to both binary and multi-class target

    Parameters
    ----------
    dataframe: DataFrame or named Series
    
    target_column: str
        Name of the target column
        
    add_prefix: Bool. Default is True
        Include a prefix of the target column in the dataset
        
    drop: Bool. Default is True
        drop original target column name
        
    display_inline:Bool. Default is  True
    
    Returns
    -------
    Pandas Dataframe:
        A new dataframe with mapped target column
    """
    if dataframe is None:
        raise ValueError("data: Expecting a DataFrame or Series, got 'None'")
    
    if not isinstance(target_column,str):
        errstr = f'The given type for column_name is {type(target_column).__name__}. Expected type is str'
        raise TypeError(errstr)

    new_data = dataframe.copy()
    data = new_data[new_data[target_column].notnull()].reset_index(drop=True)
    num_unique = data[target_column].unique()
    if new_data.shape[0] != data.shape[0]:
        if display_inline:
            print('Records with Null/Nan values in the target_column have been dropped\n')
            print(f'New data shape is {data.shape}')
    elem = data[target_column].value_counts()
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
        
    if display_inline:
        print_divider('Mapping target columns')
        for key,value in items.items():
            print(f'{key} was mapped to {value}\n')
    if add_prefix:
        prefix_name = f'transformed_{target_column}'
    else:
        prefix_name = target_column
    data.loc[:,prefix_name] = data[target_column].map(items)
    
    if drop:
        data = manage_columns(data,target_column,drop_columns=True,drop_duplicates=None)
    return data

    
def _preprocess_non_target_col(data=None,processed_data_path=None,display_inline=True,
                              select_columns=None,**kwargs):

    '''
    Automatically preprocess dataframe/file-path for a non_supervised use-case. 
    Handles missing value, Outlier treatment, and feature engineering. 
    
    Parameters
    ----------
    data: DataFrame or named Series
        Dataframe or dath path to the data
        
    processed_data_path: String
        Path to where the preprocessed data will be stored

    display_inline:Bool. Default is  True
            
    select_columns: List
        List of columns to be used
    
    Returns
    -------
    Pandas Dataframe:
        Returns a clean dataframe in the filepath

    '''                          
    
    if data is not None:
        if isinstance(data, pd.DataFrame):
            test_df = data
            if select_columns:
                test_df = manage_columns(test_df, columns=select_columns,select_columns=True,drop_duplicates=None)
        else:
            test_df = read_file(data, input_col= select_columns,**kwargs)
            
    test_df = drop_uninformative_fields(test_df,display_inline=display_inline)
    num_attributes = test_df.select_dtypes(exclude=['object', 'datetime64','bool']).columns.tolist()
    
    with HiddenPrints(): 
        #how to indicate columns with outliers 
        data = detect_fix_outliers(dataframe=test_df,num_features=num_attributes,n=3,display_inline=display_inline)
        data = handle_nan(dataframe=data,n=3,drop_outliers=False,display_inline=display_inline,thresh_x=1,thresh_y=99,**kwargs)

        for column in data.columns:
            if 'age' in column.lower():
                match = re.search(r'(.*?)[Aa]ge.*', column).group()
                age_column = str(match)
                data = bin_age(data,age_column)

        for name, dtype in data.dtypes.iteritems():
            if 'datetime64' in dtype.name:
                print_divider('Featurize Datetime columns')
                print(f'column with datetime type: [{name}]\n') 
                data = featurize_datetime(data,name,False)#generic #methods to bin

            elif 'object' in dtype.name:
                output = check_datefield(data, name)
                if output:
                    print_divider('Featurize Datetime columns')
                    print(f'Inferred column with datetime type: [{name}]\n') 
                    data = featurize_datetime(data,name,drop=False)
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
    
    if display_inline:
        print_divider('Display the preprocessed data')
        display(data.head(5))
    
    data.to_pickle(processed_data_path)
    print_divider('Preprocessed data saved')
    print(f'\nDone!. Input data has been preprocessed successfully and stored in {processed_data_path}')


def _preprocess(data=None,target_column=None,train=True,select_columns=None,\
               display_inline=True,project_path=None,**kwargs): 
    
    
    if data is None:
        raise ValueError("data: Expecting a DataFrame or Series or a data path, got None")
        
    if project_path is None:
        raise ValueError("project_path: Expecting a path to store the preprocessed data")
        
    if train:
        if target_column is None:
            raise ValueError("target_column: You need to specify a target column")
        
    if os.path.exists(project_path):
        pass
    else:
        os.mkdir(project_path) 
    project_path = os.path.join(project_path, 'data')
    if os.path.exists(project_path):
        pass
    else:
        os.mkdir(project_path)
    
    if data is not None:
        if isinstance(data, pd.DataFrame):
            train_df = data
            if select_columns:
                train_df = manage_columns(train_df,columns=select_columns,select_columns=True,drop_duplicates=None)
        else:
            train_df = read_file(data, input_col= select_columns,**kwargs)
        
    if target_column:
        row_length = train_df.shape[0]
        one_percent = 1/100 * row_length
        if train_df[target_column].nunique() > one_percent:
            if train_df[target_column].dtypes == 'float64' or train_df[target_column].dtypes == 'int':
                task = 'regression'
            else:
                print(train_df[target_column].dtypes)
                task = 'NLP'
        if train_df[target_column].nunique() < one_percent:
            task = 'classification'

    if train:        
        if task == 'classification':
            if not isinstance(target_column,str):
                errstr = f'The given type for target_column is {type(target_column).__name__}. Expected type is str'
                raise TypeError(errstr)
            print(f'\nThe task for preprocessing is {task}')

            PROCESSED_TRAIN_PATH = os.path.join(project_path, 'train_data.pkl')
            data = handle_nan(dataframe=train_df,target_name=target_column,display_inline=display_inline,n=3,**kwargs)
            data = map_target(data,target_column=target_column,drop=True,display_inline=display_inline)
            prefix_name = f'transformed_{target_column}'
            for column in data.columns:
                if 'age' in column.lower():
                    print_divider('Bucketize Age columns')
                    print(f' Inferred age column: [{column}]')
                    match = re.search(r'(.*?)[Aa]ge.*', column).group()
                    age_column = str(match)
                    data = bin_age(data,age_column)

            for name, dtype in data.dtypes.iteritems():
                if 'datetime64' in dtype.name or 'time' in name.lower() or 'date' in name.lower():
                    date_feature = True
                    output = check_datefield(data, name)
                    if output:
                        print_divider('Featurize Datetime columns')
                        print(f'column with datetime type: [{name}]\n') 
                        data = featurize_datetime(data,name,False)

                elif 'object' in dtype.name:
                    output = check_datefield(data, name)
                    if output:
                        if date_feature:
                            print(f'Inferred column with datetime type: [{name}]\n') 
                            data = featurize_datetime(data,name,False)

                        else:
                            print_divider('Featurize Datetime columns')
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

            data = drop_uninformative_fields(data,display_inline=display_inline)

            create_schema_file(data,target_column=prefix_name,project_path=project_path,\
                               display_inline=display_inline,id_column=data.columns[0])
            if display_inline:
                print_divider('Preview the preprocessed data')
                display(data.head(5))    

            data.to_pickle(PROCESSED_TRAIN_PATH)
            print_divider('Preprocessed data saved')
            print(f'\n Input data preprocessed successfully and stored in {PROCESSED_TRAIN_PATH}\n')
            
        elif task == 'clustering':
            PROCESSED_CLUSTER_PATH = os.path.join(project_path, 'preprocessed_cluster_data.pkl')
            _preprocess_non_target_col(data = train_df, PROCESSED_DATA_PATH = PROCESSED_CLUSTER_PATH,display_inline=display_inline,\
                                 select_columns=select_columns,**kwargs)

        elif task == 'regression':
            raise ValueError("task: Does not support regression tasks for now")
        else:
            raise ValueError("task: Does not support NLP tasks for now")

    else:
        PROCESSED_TEST_PATH = os.path.join(project_path, 'validation_data.pkl')
        _preprocess_non_target_col(data=train_df,PROCESSED_DATA_PATH = PROCESSED_TEST_PATH,display_inline=display_inline,\
                                 select_columns=select_columns,**kwargs)


def preprocess(data=None,target_column=None,train=False,select_columns=None,\
               display_inline=True,project_path=None,**kwargs): 
    
    """
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
        
    display_inline:Bool. Default is  True
    
    Returns
    -------
    Pandas Dataframe:
        Returns a clean dataframe in the filepath
    """
    

    if display_inline:
        _preprocess(data=data,target_column=target_column,train=train,select_columns=select_columns,\
               display_inline=display_inline,project_path=project_path,**kwargs)
    else:
        with HiddenPrints():
            _preprocess(data=data,target_column=target_column,train=train,select_columns=select_columns,\
               display_inline=display_inline,project_path=project_path,**kwargs)
    

def rename_similar_values(dataframe,column_name,cut_off=0.75,n=None):

    """
    Use Sequence Matcher to check for best "good enough" matches.
    
    Rename values based on similar matches.
    

    Parameters
    ----------
        dataframe: Pandas Series
        
        column_name: str. 
            Name of pandas column to perform operation on
            
        cut_off: int
            Possibilities that don't score at least that similar to word are ignored        
            
        n(optional): int. default 2. 
            The maximum number of close matches to return.  n must be > 0.

    
    Returns
    -------
        Pandas Dataframe.
        
    Example
    -------
    >>> pd.dataframe(["Lagos", "Lag", "Abuja", "Abuja FCT", 'Ibadan'],column=['column_name'])
    >>> Applying the function to this pandas series yields 
    
    >>> ["Lagos", "Lagos", "Abuja", "Abuja", 'Ibadan']
    
    """
    
    if dataframe is None:
        raise ValueError("data: Expecting a pandas Series, got 'None'")
    
    if not isinstance(column_name,str):
        errstr = f'The given type for column_name is {type(column_name).__name__}. Expected type is str'
        raise TypeError(errstr)
  
    if n is not None and n is not isinstance(n,int):
        errstr = f'The given type for n is {type(n).__name__}. Expected type is int'
        raise TypeError(errstr)
     
    if n is not None and n < 1:
        raise ValueError("n: n must be greater than ")
    
    df = dataframe.copy()
    l = df[column_name].tolist()
    map_dict = {}
    while l:
        word = l.pop(0)
        matches = get_close_matches(word, l, cutoff=0.70)
        if len(matches)==1:
            match = matches[0]
            map_dict[match] = word
            l.remove(match)
        elif len(matches)>=2:
            match1 = matches[0]
            match2 = matches[1]
            map_dict[match1] = word
            map_dict[match2] = word
            l.remove(match1)
            l.remove(match2)
        else:
            map_dict[word] = word
    rename_similar_values.map_dict = map_dict
    return df[column_name].replace(map_dict)


def trim_all_columns(dataframe):

    """
    Trim whitespace from ends of each value across all series in dataframe

    Parameters
    ----------
    dataframe: Pandas dataframe

    Returns
    -------
    Pandas Dataframe:
    """
    trim_strings = lambda x: x.strip() if isinstance(x, str) else x
    return dataframe.applymap(trim_strings)