from slik.loadfile import load_data
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RemoveOutliers(BaseEstimator,TransformerMixin):
    def __init__(self,features):
        self.features = features
        
    def fit(self,X):
        return self
    
    def transform(self,X):
        self.X = X
        for column in self.features:
            self.X[column] = abs(self.X[column])
            mean = self.X[column].mean()

            #calculate the interquartlie range
            q25, q75 = np.percentile(self.X[column].dropna(), 25), np.percentile(self.X[column].dropna(), 75)
            iqr = q75 - q25

                #calculate the outlier cutoff
            cut_off = iqr * 1.5
            lower,upper = q25 - cut_off, q75 + cut_off

                #identify outliers
            outliers = [x for x in self.X[column] if x < lower or x > upper]

            if column == 'loan_propensity':
                pass
            else:
                self.X[column] = self.X[column].apply(lambda x : mean if x < lower or x > upper else x)
        return self.X
    

class clean_data():
    
    '''
    This is where the data is cleaned at first. The typical issues of 
    outliers, NaN and normalization will be handled here
    
    '''
    
    def __init__(self, path=None, input_cols=None, data=None,*args,**kwargs):
        self.path = path
        self.input_cols = input_cols
        self.data = data
        if self.path:
            self.data = load_data(self.path,*args,**kwargs)
    
    
    def check_nan(self):
        
        """
        
        Function checks if NaN values are present in the dataset for both categorical
        and numerical variables
    
        
        """
        missing_values = self.data.isnull().sum()
        count = missing_values[missing_values>1]
        print('\n Features       Count of missing value')
        print('{}'.format(count))
    
    def handle_nan(self,strategy='mean',fillna='mode'):
        
        """
        
        Function handles NaN values in a dataset for both categorical
        and numerical variables
    
        Args:
            strategy: Method of filling numerical features
            fillna: Method of filling categorical features
        """
        self.num_attributes = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_attributes = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if strategy=='mean':
            for item in self.data[self.num_attributes]:
                self.data[item] = self.data[item].fillna(self.data[item].mean())
        if fillna == 'mode':
            for item in self.data[self.cat_attributes]:
                self.data[item] = self.data[item].fillna(self.data[item].value_counts().index[0])
        else:
            for item in self.data[self.num_attributes]:
                self.data[item] = self.data[item].fillna(fillna)
                
        return self.data
            
    def identify_columns(self, high_dim=100):
        
        """
        
        This funtion takes in the data, identify the numerical and categorical
        attributes and stores them in a list
        
        """
        
        self.handle_nan()
        print('\n There are {} numerical features in the data.'.format(len(self.num_attributes)))
        print('\n There are {} categorical features in the data'.format(len(self.cat_attributes)))
        print('\n --------------------------------------------')
        print('\n --------------------------------------------')
        
        self.low_cat = []
        self.hash_features = []
        for item in self.cat_attributes:
            if self.data[item].nunique() > high_dim:
                print('\n {} has a high cardinality. It has {} unique attributes'.format(item, self.data[item].nunique()))
                self.hash_features.append(item)
            else:
                self.low_cat.append(item)
                
    def __repr__(self):
        
        """
        Magic method to output the characteristics of the Preprocessing instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the cleaned data
        
        """
        return "The shape of the data is {}".format(self.data.shape)