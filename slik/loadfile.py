import pandas as pd

def read_file(file_path,input_col=None,**kwargs):
    """
    This funtion takes in a file path - CSV, excel or parquet and reads the data
    based on the input columns specified. Can only load file at a time. 
    
    Parameters
    ----------
    file_path: str/file path
        path to where data is stored.
    input_col: str
        select columns to be loaded as a pandas dataframe
    **kwargs:
        use keyword arguements from pandas read file method
        
    Returns
    -----------
    pandas Dataframe
        
    """
        
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, usecols = input_col,**kwargs)
        print('\nCSV file read sucessfully')
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
            
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path, engine = 'pyarrow', columns = input_col,**kwargs)
        print('Parquet file read sucessfully')
        data.columns = data.columns.astype(str)
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
        
    elif file_path.endswith('.xls'):
        data = pd.read_excel(file_path, usecols = input_col,**kwargs)
        print('Excel file read success')
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
    
    else:
        raise ValueError("file_path: Only supports one of ['csv','xls','parquet'] format")