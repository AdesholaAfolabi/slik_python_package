from pipeline import Build
import pickle
import pandas as pd

class Save_pipeline():
    '''
    This class would save the output of the pipeline
    object and sparse matrix and return it to the user
    
    '''
    
    def __init__(self, path, input_cols, data = None, pipeline = None):
        
        self.pipeline = pipeline
        self.data = data
        self.path = path
        self.input_col = input_cols
        
    def get_data_and_pipeline(self):
        
        """
        
        Function to obtain the pipeline object from the feature engineering class
        imported above among the libraries
                
        Args:
            None
        
        Returns:
            None
        
        """
        
        data = Build(self.path, input_cols = self.input_col)
        X, full_pipeline = data.build_object(hash_size=20, normalization = True)
        self.data = X
        self.pipeline = full_pipeline
    
        
    def save_model_and_data(self, pipeline_name =None, data_name = None): 
        
        """
        
        Function saves the KMeans and Pipeline objects
                
        Args:
            model_name to be used to save the pickle file
            pipeline name to be used to save the pickle file
        
        Returns:
            None
        
        """
        
        if data_name:
            self.df = pd.DataFrame(self.data)
            self.df.to_csv(data_name, index=False)
        else:
            pass
        
        if pipeline_name:
            pickle.dump(self.pipeline, open(pipeline_name, 'wb'))
    
        else:
            pass
        
    def complile_functions(self):
        
        """
        This funtion compiles all the other funtions created above.
        """
        
        self.get_data_and_pipeline()
        self.save_model_and_data(pipeline_name = "pipeline.pkl", data_name = "data.csv")
        
    
        
    