import pytest
import pandas as pd
import numpy as np
from slik import preprocessing


@pytest.fixture(scope='module')
def data():
    np.random.seed(1)
    df = pd.DataFrame(np.random.randint(0, 100,size=(100,2)),  columns=('Age','just_number'))
    df['Salary'] = np.random.randint(5000, 10000, size=(100,))
    df['Time'] = np.round(np.random.uniform(0.30, 60, size=(100,1)), 2)

    df1 = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                    'size': [280, 20, 60, np.NaN],
                    'language': ['En', 'En', 'En', np.NaN]})

    df2 = pd.DataFrame({'country': ['Nigeria', 'Ghana', 'USA', 'Germany'],
                        'size': [180, np.NaN, np.NaN, np.NaN]})

    df3 = pd.DataFrame({'ID': ['id1','id2','id3','id4'],
                    'size': [280, 20, 60, np.NaN],
                    'language': ['En', 'En', 'En', 'En'],
                    'label': [1,2, 1, 2]})
                    
    return df, df1, df2, df3

#dataset to testing age bin
@pytest.mark.parametrize("data_val, data_bin_val", [([2, 17, 30, 45, 99],['Toddler/Baby', 'Child', 'Young Adult', 'Mid-Age', 'Elderly'])])
def test_bin_age(data,data_val, data_bin_val):
    table, df1, df2, df3 = data
    look_up = dict(zip(data_val,data_bin_val))
    #testing when add_prefix is True
    add_prefix=True
    temp_df = preprocessing.bin_age(dataframe = table, age_col = 'Age', add_prefix=add_prefix)
    filtered_age = temp_df[temp_df['Age'].isin(data_val)]
    idx = filtered_age['Age'].index
    i = 0
    for key in filtered_age['Age']:  
        index = idx[i]
        output = filtered_age['transformed_Age'][index]
        assert output == look_up[key], f'Expected bin_age value {look_up[key]},but got {output}'
        i+=1
    
    add_prefix=False
    temp_df = preprocessing.bin_age(dataframe = table, age_col = 'Age', add_prefix=add_prefix)
    filtered_age = temp_df[temp_df['Age'].isin(data_val)]
    idx = filtered_age['Age'].index
    i = 0
    for key in filtered_age['Age']:  
        index = idx[i]
        output = filtered_age['Age'][index]
        assert output == look_up[key], f'Expected bin_age value {look_up[key]},but got {output}'
        i+=1


def test_drop_uninformative_fields(data):
    table, df1, df2, df3 = data
    expected = ['country', 'size']
    output = list(preprocessing.drop_uninformative_fields(df1).columns)
    assert expected == output