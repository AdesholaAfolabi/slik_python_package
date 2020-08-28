import pandas as pd


def load_data(path,*args,**kwargs)->pd.DataFrame:

    """
    Read a comma-separated values (csv) file, Excel file and a Parquet file into DataFrame.

    Returns:
        A pandas Dataframe

    """
    input_col = None
    if path.endswith('.csv'):
        data = pd.read_csv(path, *args,**kwargs)
        print('CSV file read sucessfully')
        data = data.reindex(columns = input_col)
        return data

    elif path.endswith('.parquet'):
        data = pd.read_parquet(path, engine = 'pyarrow',*args,**kwargs)
        print('Parquet file read sucessfully')
        data.columns = data.columns.astype(str)
        data = data.reindex(columns = input_col)
        return data

    elif path.endswith('.xls'):
        data = pd.read_excel(path, *args,**kwargs)
        print('Excel file read success')
        data = data.reindex(columns = input_col)
        return data

    else:
        return ('No CSV file or Parquet file or Excel file was passed')

