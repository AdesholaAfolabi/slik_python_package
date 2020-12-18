import pandas as pd

def read_file(file_path,input_col=None,**kwargs):
    """
        
        This funtion takes in a file path - CSV, excel or parquet and reads the data
        based on the input columns specified 
        
        Returns:
            dataset to be used for training
        
    """
        
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path, usecols = input_col)
        print('\nCSV file read sucessfully')
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
            
    elif file_path.endswith('.parquet'):
        data = pd.read_parquet(file_path, engine = 'pyarrow', columns = input_col)
        print('Parquet file read sucessfully')
        data.columns = data.columns.astype(str)
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
        
    elif file_path.endswith('.xls'):
        data = pd.read_excel(file_path, usecols = input_col)
        print('Excel file read success')
        data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data

    elif file_path.endswith('.json'):
        data = pd.read_json(file_path, usecols = input_col)
        print('Excel file read success')
        #data = data.reindex(columns = input_col)
        print ('\nData has {} rows and {} columns'.format(data.shape[0],data.shape[1]))
        return data
        
    else:
        return ('No CSV file or Parquet file or Excel file was passed')