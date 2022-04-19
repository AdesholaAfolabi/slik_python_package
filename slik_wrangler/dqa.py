"""
High level support for performing assessment on the quality of datasets
"""

import numpy as np

from .messages import log
from .preprocessing import check_nan

from IPython.display import display


def missing_value_assessment(dataframe, display_findings=True):
    """
    Performs assessment of missing values in a dataframe.
    
    Function performs an assessment of the missing values from any given dataframe 
    and generates a report of its findings if requested.
    
    Parameters:
    ------------
    dataframe: Pandas Dataframe
        This is the dataset to be evaluated for missing values.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the missing values and percentage of missing values.
    """
    
    check_nan(dataframe, display_inline=False)
    
    df = check_nan.df
    df.set_index('features', inplace=True)
    df = df[df.missing_counts > 0]
    
    if len(df):
        log(
            f"Dataframe contains missing values that you should address. \n\ncolumns={list(df.index)}\n", 
            code='warning'
        )
        
        if display_findings:
            display(df)
    else:
        log("No missing values!!!", code='success')


def duplicate_assessment(dataframe, display_findings=True):
    """
    Performs assessment of duplicate values in a dataframe.
    
    Function performs an assessment of the duplicate values from any given dataframe 
    and generates a report of its findings if requested. It does this assessment for 
    both rows and feature columns.
    
    Parameters:
    ------------
    dataframe: pandas Dataframe
        This is the dataset to be evaluated for duplicate values.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the duplicated row and columns.
    """
    
    def check_duplicate_columns(df):
        """
        Checks for all the duplicates columns in the dataset
        """
        
        duplicated = set()
        column_names = list(df.columns)
        
        def check(col1, col2):
            """
            Checks if the two columns are identical
            """
            
            col1, col2 = list(map(
                lambda x: df[x].to_numpy().ravel(), (col1, col2)
            ))
            
            return (col1 == col2).all()
        
        for col in column_names:
            if col not in duplicated:
                _col = column_names.copy()
                _col.pop(_col.index(col))

                for c in _col:
                    if c.lower() == col.lower():
                        log("Two columns bears identical names", c, "address this problem", code='danger')
                    elif check(c, col):
                        duplicated.add(c, col)
        
        return list(duplicated)
    
    duplicated_rows = dataframe[dataframe.duplicated()]
    duplicated_columns = check_duplicate_columns(dataframe)
    
    if len(duplicated_rows):
        log(
            f"Dataframe contains duplicate rows that you should address. \n\ncolumns={list(dupulicated_rows.index)}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dupulicated_rows)
            
    if len(duplicated_columns):
        log(
            f"Dataframe contains duplicate columns that you should address. \n\ncolumns={duplicated_columns}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dataframe[duplicated_columns])
    
    if not len(duplicated_rows) and not len(duplicated_columns):
        log("No duplicate values in both rows and columns!!!", code='success')


def structure_assessement(dataframe, display_findings=True):
    """
    Checks the data structure of each feature column.
    
    Function checks if the data type of each feature column is consistent.
    e.g. if there is an interger variable and a string variable within a
    feature column or more.
    
    For categorical columns. Assessment is conducted on similar categories
    e.g. medium and Medium are the same and one have to be changed 
    for the other, so it is either medium or Medium.
    
    This errors are mainly generated during data collection.
    
    Parameters:
    ------------
    dataframe: pandas Dataframe
        This is the dataset to be evaluated for consistent data type.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the feature columns with poor structures.
    """
    
    column_names = list(dataframe.columns)
    inconsistent_cols = []
    
    def check_consistent_in_type(_col):
        """
        Checks for consistency in the data type
        """
        
        return all([type(c) == _col.dtype for c in _col.to_numpy()])
    
    def check_consistent_in_categories(_col):
        """
        Checks for consistency in the categories
        """
        
        unique_str_cat = [c for c in _col.unique() if type(c) == str]
        
        categories = list(map(lambda x: x.lower(), unique_str_cat))
        unique_categories = set(categories)
        
        for cat in unique_categories:
            if categories.count(cat) > 1:
                return False
        
        return True
    
    for col in column_names:
        col_data = dataframe[col]
        
        if col_data.dtype == np.object0:
            check_result = not (
                check_consistent_in_type(col_data) or 
                check_consistent_in_categories(col_data)
            )
        else:
            check_result = not check_consistent_in_type(col_data)
        
        if check_result:
            inconsistent_cols.append(col)
    
    if len(inconsistent_cols):
        log(
            f"Dataframe contains inconsistent feature columns that you should address. \n\ncolumns={inconsistent_cols}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dataframe[inconsistent_cols].head())
    else:
        log("No inconsistent feature columns values!!!", code='success')


def data_cleanness_assessment(dataframe, display_findings=True):
    """
    Checks for the overall cleanness of the dataframe by:
    
    1. Checking if there are missing values in the dataset (dataframe)
    2. Checking if the dataset (dataframe) contains any duplicates
    3. checking if there are any poor structure among feature columns
    
    and finally a report is generated based on it's findings if requested
    
    Parameters:
    ------------
    dataframe: pandas Dataframe
        This is the dataset to be evaluated on how clean it is.
        
    display_findings: boolean, Default True
        Whether or not to display dataframes highlighting
        the observations from checking for missing values, 
        duplicate values and poor structures.
    """
    
    issue_checker = {
        'missing values': missing_value_assessment,
        'duplicate variables': duplicate_assessment,
        'poor structure': structure_assessement
    }
    
    for issue in issue_checker.keys():
        log(f"Checking for {issue}", end="\n\n", code='info')
        issue_checker[issue](dataframe, display_findings)
        log(end="\n\n")
