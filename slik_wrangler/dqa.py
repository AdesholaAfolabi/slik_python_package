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
    
    results = list(results)
    
    f = limit // 2
    ff = f + limit % 2
    
    results = results[:ff] + ['...'] + results[-f:] if len(results) > limit else results
    
    result = '[' + ', '.join(map(str, results)) + ']'
    
    return result


def missing_value_assessment(dataframe, display_findings=True, return_as_str=False):
    """
    Checks the missing values from the given datset and generates
    a report of its findings.
    
    dataframe: pandas Dataframe
        Data set to perform assessment on.
        
    display_findings: boolean, Default True
        Whether or not to display a dataframe highlighting
        the missing values count and percentage.
    """
    
    df = check_nan(dataframe, display_inline=True)
    # print(df)
    
    # df = check_nan.df
    df.set_index('features', inplace=True)
    df = df[df.missing_counts > 0]
    
    missing_columns = list(df.index)
    
    missing_value_assessment.missing_columns = missing_columns
    
    if len(df):
        message = (
            f"\nDataframe contains missing values that you should address. "
            f"\n\ncolumns={__summarise_results(missing_columns)}\n",
            df
        )

        log(message[0], code='warning')
        
        if display_findings:
            return df
    else:
        message = ("No missing values!!!",)

        log(message[0], code='success')

    if return_as_str:
        return message


def duplicate_assessment(dataframe, display_findings=True, return_as_str=False):
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
    
    transfromed_dataframe = dataframe.T.copy()
    
    duplicated_columns = transfromed_dataframe[transfromed_dataframe.duplicated()]
    duplicated_rows = dataframe[dataframe.duplicated()]
    
    duplicate_assessment.duplicated_columns = list(duplicated_columns.index)
    duplicate_assessment.duplicated_rows = list(duplicated_rows.index)

    message = ()
    
    if len(duplicated_columns):
        log_message = f"Dataframe contains duplicate columns that you should address. " \
                      f"\n\ncolumns={__summarise_results(list(duplicated_columns.index))}\n"

        message += log_message,
        log(log_message, code='warning')
        
        if display_findings:
            display(duplicated_columns.T)
    
    if len(duplicated_rows):
        log_message = f"Dataframe contains duplicate rows that you should address. " \
                      f"\n\nrows={__summarise_results(list(duplicated_rows.index))}\n"

        message += log_message,
        log(log_message, code='warning')
        
        if display_findings:
            display(duplicated_rows)
    
    if not len(duplicated_rows) and not len(duplicated_columns):
        log_message = "No duplicate values in both rows and columns!!!"
        message += log_message,
        log(log_message, code='success')

    if return_as_str:
        return message
        
        
def outliers_assessment(dataframe, display_findings=True, return_as_str=False):
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
    
    outliers_assessment.contains_outliers = contains_outliers
    
    if len(contains_outliers):
        message = (
            "Ignore target column, if target column is considered an outlier\n",
            f"Dataframe contains outliers that you should address. "
            f"\n\ncolumns={__summarise_results(contains_outliers)}\n",
            dataframe[contains_outliers]
        )

        log(message[0], code="info")
        log(message[1], code='warning')
        
        if display_findings:
            display(dataframe[contains_outliers].head())
    else:
        message = ("No outliers in dataset!!!",)

        log(message[0], code='success')

    if return_as_str:
        return message


def consistent_structure_assessement(dataframe, display_findings=True, return_as_str=False):
    """
    Checks the consitent nature of each feature column.
    
    It checks if the dtype across each feature column is consistent.
    i.e. if there is an interger variable and a string variable across
    the various feature columns.
    
    For categorical columns. assessment is made on duplicate categories
    i.e. medium and Medium are the same categories and inconsistent.
    """
    
    def check_consistent_in_type(_col):
        """
        Checks for consistency in the dtype
        """
        
        return all(map(lambda c: type(c) == _col.dtype, _col.to_numpy()))
    
    def check_consistent_in_categories(_col):
        """
        Checks for consistency in the categories
        
        For categorical columns
        """
        
        unique_str_cat = filter(lambda c: type(c) == str, _col.unique())
        
        categories = map(lambda x: x.lower(), unique_str_cat)
        unique_categories = set(categories)
        
        for cat in unique_categories:
            if list(categories).count(cat) > 1:
                return False
        
        return True
    
    column_names = list(dataframe.columns)
    inconsistent_cols = []
    
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
    
    consistent_structure_assessement.inconsistent_cols = inconsistent_cols
    
    if len(inconsistent_cols):
        message = (
            f"Dataframe contains inconsistent feature columns that you should address. "
            f"\n\ncolumns={__summarise_results(inconsistent_cols)}\n",
            dataframe[inconsistent_cols]
        )

        log(message[0], code='warning')
        
        if display_findings:
            display(dataframe[inconsistent_cols].head())
    else:
        message = ("No inconsistent feature columns values!!!",)
        log(message[0], code='success')

    if return_as_str:
        return message


def data_cleanness_assessment(dataframe, display_findings=True, return_as_str=False):
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

    if return_as_str:
        message = ()
        for issue in issue_checker.keys():
            message += f"**Checking for {issue}**",
            message += '\n\n',
            message += issue_checker[issue](dataframe, display_findings, return_as_str)
            message += '\n\n',

        return message
    else:
        for issue in issue_checker.keys():
            log(f"Checking for {issue}", end="\n\n", code='info')
            issue_checker[issue](dataframe, display_findings)
            log(end="\n\n")



