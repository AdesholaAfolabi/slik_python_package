from general_utils import Read_file
import pandas as pd
import numpy as np


class Clean(Read_file):
    
    '''
    This is where the data is cleaned at first. The typical issues of 
    outliers, NaN and normalization will be handled here
    
    '''
    
    def __init__(self, path, input_cols):
        
        Read_file.__init__(self, path, input_cols)
        
    def identify_columns(self):
        
        """
        
        This funtion takes in the data, identify the numerical and categorical
        attributes and stores them in a list
        
        """
        
        self.num_attributes = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_attributes = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    def handle_nan(self):
        
        """
        
        Function handles NaN values in a dataset for both categorical
        and numerical variables
    
        
        """
        
        for item in self.data[self.num_attributes]:
            self.data[item] = self.data[item].fillna(self.data[item].mean())
        for item in self.data[self.cat_attributes]:
            self.data[item] = self.data[item].fillna(self.data[item].value_counts().index[0])
            
    def identify_cat(self):
        
        """
        
        This function will identify categorical columns with low and high dimensionality automatically
        
        """
        self.read_data()
        self.identify_columns()
        self.handle_nan()
        self.low_cat = []
        self.hash_features = []
        
        for item in self.cat_attributes:
            if len(self.data[item].value_counts()) > 10:
                self.hash_features.append(item)
            else:
                self.low_cat.append(item)
                
        print (self.hash_features)