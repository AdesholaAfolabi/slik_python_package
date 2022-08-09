"""high level support for loading files."""

import pandas as pd
import csv
import os
from .messages import log


def read_file(file_path, input_col=None, **kwargs):

    """
    Load a file path into a dataframe.
    
    This funtion takes in a file path - CSV, excel or parquet and reads 
    the data based on the input columns specified. Can only load  one file at 
    a time. 
    
    Parameters
    ----------
    file_path: str/file path
        path to where data is stored.
    input_col: list
        select columns to be loaded as a pandas dataframe
    **kwargs:
        use keyword arguements from pandas read file method
        
    Returns
    -----------
    pandas Dataframe
        
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, usecols= input_col,**kwargs)
        log('\nCSV file read sucessfully', code='success')
        data = data.reindex(columns = input_col)
        log('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]), code='info')
        return data
            
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path, engine = 'pyarrow', columns = input_col,**kwargs)
        log('Parquet file read sucessfully', code='success')
        data.columns = data.columns.astype(str)
        data = data.reindex(columns = input_col)
        log('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]), code='info')
        return data
        
    elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path, usecols = input_col,**kwargs)
        log('Excel file read successfully', code='success')
        data = data.reindex(columns = input_col)
        log('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]), code='info')
        return data
    
    elif file_path.endswith('.json'):
        data = pd.read_json(file_path, usecols = input_col)
        log('JSON file read successfully', code='success')
        data = data.reindex(columns = input_col)
        log('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]), code='info')
        return data
        
    else:
        raise ValueError("file_path: Only supports one of ['csv','xls','xlsx','parquet'] format")


def split_csv_file(file_path=None, delimiter= ',' , row_limit=1000000, output_path='.', keep_headers=True):

    """
    Split large csv files to small csv files.

    Function splits large csv files into smaller files based on the row_limit
    specified. The files are stored in present working dir by default.
    
    Parameters
    ----------
    file_path: str/file path
        path to where data is stored.
    delimiter: str. Default is ','
        separator in each row and column,
    row_limit: int
        split each file by row count
    output_path: str
        output path to store splitted files
    keep_headers: Bool. Default is True
        make use of headers for all csv files
        
    Returns
    -----------
    Splitted files are stored in output_path
    """
    fp = open(file_path, 'r')
    output_name_template='output_%s.csv'
    reader = csv.reader(fp, delimiter=delimiter)
    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    for i, row in enumerate(reader):
        if i + 1 > current_limit:
            current_piece += 1
            current_limit = row_limit * current_piece
            current_out_path = os.path.join(
                output_path,
                output_name_template % current_piece
            )
            current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
            if keep_headers:
                current_out_writer.writerow(headers)
        current_out_writer.writerow(row)
