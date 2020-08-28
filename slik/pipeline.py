from slik.preprocessing import clean_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split


class build_pipeline(clean_data):
    
    '''
    This class is where the building of the pipeline
    object to be used for modeling takes place
    
    '''
    
    def __init__(self, path, input_cols):
        
        clean_data.__init__(self, path, input_cols)
        
    
    def pipeline(self, hash_size, scaling = None, normalization = None):
        
        """
        
         Function contains the pipeline methods to be used.
         It is broken down into numerical, categorical and hash pipeline
                
        """
        if scaling:
            
            self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', StandardScaler())])
            self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                       ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
            self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
            
        elif normalization:
            
            self.num_pipeline = Pipeline(steps= [('imputer', SimpleImputer(strategy='mean')), ('std_scaler', MinMaxScaler())])
            self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                       ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
            self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
            
        else:
            
            self.cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                       ('one_hot_encoding', OneHotEncoder(handle_unknown = "ignore", sparse = False))])
            self.hash_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                                  ('hasher', FeatureHasher(n_features=hash_size, input_type='string'))])
    
    
    def build_object(self, hash_size, scaling = None, normalization =None):
        
        """
        
        Function that builds the pipeline and returns the 
        pipeline object and the data to be used for modeling
                
        Args:
            hash_bucket size, scaling, normalization
        
        Returns:
            pipeline object
            data to be used for training after being transformed by the pipeline
        
        """
        
        self.identify_cat()
        self.pipeline(hash_size, scaling, normalization)
        self.full_pipeline = ColumnTransformer(
        transformers=[
            ('num', self.num_pipeline, self.num_attributes),
            ('cat', self.cat_pipeline, self.low_cat),
            ('hash', self.hash_pipeline, self.hash_features)
        ])
        
        self.X = self.data
        
        self.full_pipeline.fit(self.X)
        
        self.X = self.full_pipeline.transform(self.X)
        
        print(self.X.shape)
        return self.X, self.full_pipeline
        


