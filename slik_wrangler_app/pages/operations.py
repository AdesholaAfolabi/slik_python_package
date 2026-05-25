"""
Package encapsulates all the function responsible for
handling the webapps operations.

Format should follow:

def perform_{operation-name}_operation(dataframe):
    >>> ["CODE"]
    return changed_dataframe
"""

import sys
import numpy as np
import docs

import streamlit as st
from utils import st_stdout

from slik_wrangler import dqa
from slik_wrangler import preprocessing as pp


# Functions
def perform_bin_age_operation(dataframe):
    """
    Performs operations on binning columns

    :param dataframe: Dataframe to work with
    :return: dataframe with binned age column if selected
    """

    st.markdown("### Performing operation on BIN FIELD")
    choice = st.sidebar.selectbox('select age column', list(dataframe.describe().columns))
    add_prefix = st.sidebar.radio("add prefix", (True, False), horizontal=True)

    return (
        pp.bin_age,
        {'dataframe': dataframe, 'age_col': choice, 'add_prefix': add_prefix}
    )


def perform_check_nan_operation(dataframe):
    """
    Checks for missing values in the dataframe in two methods:

    1. As a plot function
    2. As a Dataframe indicating the missing value count
    """

    view_toggle = st.sidebar.radio('Choose view', ["Plot", "Dataframe"])

    with st_stdout("info"):
        if view_toggle == 'Plot':
            pp.check_nan(dataframe, plot=True, display_inline=True)
        else:
            df = pp.check_nan(dataframe, display_inline=True)
            df = df[df.missing_counts > 0]
            st.dataframe(df)


def perform_date_field_operation(dataframe):
    """
    Asserts if a selected field is a date-field.
    """

    st.markdown("### Performing operation on DATE FIELD")
    choice = st.sidebar.selectbox('select column', list(dataframe.columns), key='date')
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(f'###### Asserting {choice} is a date field:')
        st.write(pp.check_datefield(dataframe, choice))


def perform_dui_operation(dataframe):
    """
    Performs the operation of dropping uninformative columns
    """

    with st_stdout("info"):
        index = len(dataframe.columns) - 1
        choice = st.sidebar.selectbox(
            'exclude columns',
            list(dataframe.columns),
            key='dui',
            help="""columns to be excluded from being dropped""",
            index=index
        )

    return (
        pp.drop_uninformative_fields,
        {'dataframe': dataframe, 'exclude': choice}
    )


def perform_dqa_operation(dataframe):
    """
    Performs Data Quality Assessment Operations on the Dataset
    """

    st.markdown("### Performing operation on DQA")
    choice = st.sidebar.radio('quality assessment', ['missing value', 'duplicate assessment'])

    with st_stdout("info"):
        if choice == 'missing value':
            st.write(dqa.missing_value_assessment(dataframe))
        else:
            docs.section_not_available(add_plain_image=True)


def perform_featurize_dt_operation(dataframe):
    """
    Checks date-fields and performs operation on the dates as ordered.
    """

    choice = st.sidebar.selectbox('select date/datetime column', dataframe.columns.tolist())

    if choice:
        date_feat = [
            'Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Week',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Hour', 'Minute',
            'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Date'
        ]

        feat_choice = st.sidebar.multiselect('select a date feature', date_feat)

        drop = st.sidebar.radio("drop original datetime column", (True, False), horizontal=True)

        dataframe = pp.featurize_datetime(dataframe, choice, feat_choice, drop)
        with st_stdout("info"):
            st.dataframe(dataframe)

        return (
            pp.featurize_datetime,
            {
                'dataframe': dataframe, 'column_name': choice,
                'date_features': feat_choice, 'drop': drop
            }
        )


def perform_get_attr_operation(dataframe):
    index = len(dataframe.columns) - 1

    choice = st.sidebar.selectbox(
        'select target column',
        list(dataframe.columns),
        key='get_attr',
        help="""target column to be excluded""",
        index=index
    )

    num_feat, cat_feat = pp.get_attributes(dataframe, choice)

    with st_stdout("info"):
        st.write({
            "Numerical Features": num_feat,
            "Categorical Features": cat_feat
        })


def perform_manage_col_operation(dataframe):
    """
    Performs operation to manage the columns and features in a dataframe.
    """

    manage_sel = st.sidebar.radio(
        "Manage columns",
        ("select columns", "drop columns")
    )

    drop_dupl_col = st.sidebar.checkbox("drop duplicates")

    if manage_sel == 'drop columns':
        drop_columns = True
        choice = st.sidebar.multiselect('drop columns', dataframe.columns.tolist())
        transformed_df = pp.manage_columns(dataframe, columns=choice, drop_columns=drop_columns)
    else:
        drop_columns = False
        choice = st.sidebar.multiselect('select columns', dataframe.columns.tolist())
        transformed_df = pp.manage_columns(dataframe, columns=choice, select_columns=True)

    if drop_dupl_col:
        choice = st.sidebar.selectbox('drop duplicates across', ['rows', 'columns', 'both'])
        with st_stdout("info"):
            st.dataframe(pp.manage_columns(transformed_df, drop_duplicates=choice))

    return (
        pp.manage_columns,
        {
            'dataframe': dataframe, 'columns': choice,
            'drop_columns': drop_columns, 'select_columns': not drop_columns
        }
    )


def perform_detect_fix_outliers_operation(dataframe):
    """
    Fixes outliers from the selected feature columns.
    """

    col_list = st.sidebar.multiselect('select feature(s) to fix outliers', list(dataframe.columns))
    index = len(col_list) - 1
    choice = st.sidebar.selectbox(
        'select target column', list(dataframe.describe().columns), key='get_attr',
        help="""target column to be excluded""", index=index
    )

    n = st.sidebar.number_input("Minimum number of outliers", 1)
    num_feat = dataframe.select_dtypes(include=np.number).columns.tolist()
    fix_method = st.sidebar.selectbox("Outlier fix method", ['mean', 'log_transformation'])
    display_inline = st.sidebar.radio('Inline display', ['False', 'True'])

    with st_stdout("info"):
        dataframe = pp.detect_fix_outliers(
            dataframe, n=n, num_features=num_feat,
            fix_method=fix_method, display_inline=display_inline
        )

    return (
        pp.detect_fix_outliers,
        {
            'dataframe': dataframe, 'n': n, 'num_features': num_feat,
            'fix_method': fix_method, 'display_inline': display_inline
        }
    )
