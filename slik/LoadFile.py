import pandas as pd

class read_file():
    
    '''
    
    This class will read in the file from the user and 
    take into account whatever input columns are specified.
    
    '''
    
    def __init__(self, path, input_cols):
        
        self.path = path
        self.input_col = input_cols
        self.data = pd.DataFrame()
        
    def read_data(self):->pd.Dataframe
        
        """
        Read a comma-separated values (csv) file, Excel file and a Parquet file into DataFrame.
        
        Returns:
            A pandas Dataframe
        
        """
        
        if self.path.endswith('.csv'):
            data = pd.read_csv(self.path, usecols = self.input_col)
            print('CSV file read sucessfully')
            data = data.reindex(columns = self.input_col)
            self.data = data
            return data
            
        elif self.path.endswith('.parquet'):
            data = pd.read_parquet(self.path, engine = 'pyarrow', columns = self.input_col)
            print('Parquet file read sucessfully')
            data.columns = data.columns.astype(str)
            data = data.reindex(columns = self.input_col)
            self.data = data
            return self.data
        
        elif self.path.endswith('.xls'):
            data = pd.read_excel(self.path, usecols = self.input_col)
            print('Excel file read success')
            data = data.reindex(columns = self.input_col)
            self.data = data
            return self.data
        
        else:
            return ('No CSV file or Parquet file or Excel file was passed')
    