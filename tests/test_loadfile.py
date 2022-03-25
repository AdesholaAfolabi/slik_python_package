import sys
sys.path.insert(0, '/Users/afolabiadeshola/Downloads/slik_python_package/')


import pandas as pd
from slik.loadfile import read_file

import pytest 
import sys,os
from os.path import dirname

# spath = os.path.join(dirname(os.getcwd()), 'slik_python_package')
# sys.path.insert(0, spath)

path = os.path.join(dirname(os.getcwd()), 'examples/titanic.csv')
# sys.path.insert(0, path)

@pytest.mark.parametrize("file_path, input_col", [(path, ['PassengerId' ,'Survived' ,'Name' ,'Sex' ,'Age'])])
def test_read_file(file_path,input_col):
    dataframe = read_file(path)
    assert isinstance(dataframe,pd.DataFrame), "Not one of ['csv','xlsx','parquet'] file type"

