"""
Module for Asseting the Data Quality
"""

import numpy as np

from .messages import log
from .utils import print_divider
from .preprocessing import check_nan, get_attributes

from IPython.display import display


def __summarise_results(results, limit=10):
    """
    Hidden Function
    
    Computes a summary of results given a list object
    """
    
    assert type(results) == list, "Results must be a list object"
    
    f = limit // 2
    ff = f + limit % 2
    
    results = results[:ff] + ['...'] + results[-f:] if len(results) > limit else results
    
    result = '[' + ', '.join(map(str, results)) + ']'
    
    return result


def missing_value_assessment(dataframe, display_findings=True):
    """
    Checks the missing values from the given datset and generates
    a report of its findings.
    
    dataframe: pandas Dataframe
        Data set to perform assessment on.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the missing values count and percentage.
    """
    
    check_nan(dataframe, display_inline=False)
    
    df = check_nan.df
    df.set_index('features', inplace=True)
    df = df[df.missing_counts > 0]
    
    if len(df):
        log(
            f"Dataframe contains missing values that you should address. \n\ncolumns={__summarise_results(list(df.index))}\n", 
            code='warning'
        )
        
        if display_findings:
            display(df)
    else:
        log("No missing values!!!", code='success')


def duplicate_assessment(dataframe, display_findings=True):
    """
    Checks the duplicate values from the given datset and generates
    a report of its findings. It does this assessment for both rows
    and feature columns.
    
    dataframe: pandas Dataframe
        Data set to perform assessment on.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the missing values count and percentage.
    """
    
    def check_duplicate_columns(df):
        """
        Checks for all the duplicates rows in the dataset
        """
        
        duplicated = set()
        column_names = list(df.columns)
        
        def check(col1, col2):
            """
            Checks the equality between columns
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
                        duplicated.add(c)
                        duplicated.add(col)
        
        return list(duplicated)
    
    duplicated_rows = dataframe[dataframe.duplicated()]
    duplicated_columns = check_duplicate_columns(dataframe)
    
    if len(duplicated_rows):
        log(
            f"Dataframe contains duplicate rows that you should address. \n\nrows={__summarise_results(list((duplicated_rows.index)))}\n", 
            code='warning'
        )
        
        if display_findings:
            display(duplicated_rows)
            
    if len(duplicated_columns):
        log(
            f"Dataframe contains duplicate columns that you should address. \n\ncolumns={__summarise_results(duplicated_columns)}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dataframe[duplicated_columns])
    
    if not len(duplicated_rows) and not len(duplicated_columns):
        log("No duplicate values in both rows and columns!!!", code='success')
        
        
def outliers_assessment(dataframe, display_findings=True):
    """
    Checks for outliers in the given datset and generates
    a report of its findings.
    
    dataframe: pandas Dataframe
        Data set to perform assessment on.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the missing values count and percentage.
    """
    
    num_attributes, _ = get_attributes(dataframe)
    
    def contains_outliers(column):
        """
        Checks if the given column contains outliers
        """
        
        dataframe.loc[:,column] = abs(dataframe[column])
        
        q25 = np.percentile(dataframe[column].dropna(), 25)
        q75 = np.percentile(dataframe[column].dropna(), 75)

        outlier_cut_off = ((q75 - q25) * 1.5)
        lower_bound, upper_bound = (q25 - outlier_cut_off), (q75 + outlier_cut_off)

        outlier_list_col = dataframe[column][(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)].index
        
        return bool(len(outlier_list_col))
    
    contains_outliers = [column for column in num_attributes if contains_outliers(column)]
    
    if len(contains_outliers):
        log(
            f"Ignore target column, if target column is considered an outlier\n",
            code="info"
        )
        log(
            f"Dataframe contains outliers that you should address. \n\ncolumns={__summarise_results(contains_outliers)}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dataframe[contains_outliers].head())
    else:
        log("No outliers in dataset!!!", code='success')


def consistent_structure_assessement(dataframe, display_findings=True):
    """
    Checks the consitent nature of each feature column.
    
    It checks if the dtype across each feature column is consistent.
    i.e. if there is an interger variable and a string variable across
    the various feature columns.
    
    For categorical columns. assessment is made on duplicate categories
    i.e. medium and Medium are the same categories and inconsistent.
    """
    
    column_names = list(dataframe.columns)
    inconsistent_cols = []
    
    def check_consistent_in_type(_col):
        """
        Checks for consistency in the dtype
        """
        
        return all([type(c) == _col.dtype for c in _col.to_numpy()])
    
    def check_consistent_in_categories(_col):
        """
        Checks for consistency in the categories
        
        For categorical columns
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
            f"Dataframe contains inconsistent feature columns that you should address. \n\ncolumns={__summarise_results(inconsistent_cols)}\n", 
            code='warning'
        )
        
        if display_findings:
            display(dataframe[inconsistent_cols].head())
    else:
        log("No inconsistent feature columns values!!!", code='success')


def data_cleanness_assessment(dataframe, display_findings=True):
    """
    Checks for the overall cleanness of the dataframe:
    
    1. Checks if there are missing values in the dataset
    2. Checks if the dataset set contains any duplicates
    3. checks if there are any inconsistent feature columns
    
    Gives a report.
    """
    
    issue_checker = {
        'missing values': missing_value_assessment,
        'duplicate variables': duplicate_assessment,
        'outliers': outliers_assessment,
        'inconsistent values': consistent_structure_assessement
    }
    
    for issue in issue_checker.keys():
        log(f"Checking for {issue}", end="\n\n", code='info')
        issue_checker[issue](dataframe, display_findings)
        log(end="\n\n")

