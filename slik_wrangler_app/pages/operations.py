import sys
import docs

import streamlit as st
#from streamlit.scriptrunner import script_run_context
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx


from io import StringIO
from utils import _divider
from threading import current_thread
from contextlib import contextmanager

from slik_wrangler import dqa
from slik_wrangler import preprocessing as pp


# Wrappers
def operation_on(operation):
    """

    :param operation: Operation function or subroutine to wram
    :return:
    """

    def __internal(dataframe):
        operation(dataframe)
        _divider()

    return __internal


@contextmanager
def st_redirect(src, dst, how=None):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    s = []

    with StringIO() as buffer:
        old_write = src.write

#script_run_context
        def new_write(b):
            if getattr(current_thread(), get_script_run_ctx.SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                value = buffer.getvalue()
                s.append(value)
                if not how:
                    output_func(value)
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

    if how == 'json':
        st.json(s[0])


@contextmanager
def st_stdout(dst, how=None):
    with st_redirect(sys.stdout, dst, how):
        yield


# Functions
@operation_on
def perform_bin_age_operation(dataframe):
    """

    :param dataframe:
    :return:
    """

    st.markdown("### Performing operation on BIN FIELD")
    choice = st.sidebar.selectbox('select age column', list(dataframe.columns))
    add_prefix = st.sidebar.radio("add prefix", (True, False), horizontal=True)

    return pp.bin_age(dataframe, choice, add_prefix)


@operation_on
def perform_check_nan_operation(dataframe):
    view_toggle = st.sidebar.radio('Choose view', ["Plot", "Dataframe"])

    with st_stdout("info"):
        if view_toggle == 'Plot':
            pp.check_nan(dataframe, plot=True, streamlit=True)
        else:
            df = pp.check_nan(dataframe, display_inline=True)
            df = df[df.missing_counts > 0]
            st.dataframe(df)


@operation_on
def perform_date_field_operation(dataframe):
    """

    :param dataframe:
    :return:
    """

    st.markdown("### Performing operation on DATE FIELD")
    choice = st.sidebar.selectbox('select column', list(dataframe.columns), key='date')
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(f'###### Asserting {choice} is a date field:')
        st.write(pp.check_datefield(dataframe, choice))


@operation_on
def perform_dui_operation(dataframe):
    with st_stdout("info"):
        index = len(dataframe.columns) - 1
        choice = st.sidebar.selectbox(
            'exclude columns',
            list(dataframe.columns),
            key='dui',
            help="""columns to be excluded from being dropped""",
            index=index
        )
    return pp.drop_uninformative_fields(dataframe, exclude=choice)


@operation_on
def perform_dqa_operation(dataframe):
    st.markdown("### Performing operation on DQA")
    choice = st.sidebar.radio('quality assessment', ['missing value', 'duplicate assessment'])
    with st_stdout("info"):
        if choice == 'missing value':
            st.write(dqa.missing_value_assessment(dataframe))
        else:
            docs.section_not_available(add_plain_image=True)


@operation_on
def perform_featurize_dt_operation(dataframe):
    choice = st.sidebar.selectbox('select date/datetime column', dataframe.columns.tolist())

    if choice:
        date_feat = [
            'Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Week',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Hour', 'Minute',
            'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Date'
        ]

        feat_choice = st.sidebar.multiselect('select a date feature', date_feat)

        drop = st.sidebar.radio("drop original datetime column", (True, False), horizontal=True)

        with st_stdout("info"):
            st.dataframe(pp.featurize_datetime(dataframe, choice, feat_choice, drop))


@operation_on
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
        st.write({"Numerical Features": num_feat, "Categorical Features": cat_feat})


@operation_on
def perform_manage_col_operation(dataframe):
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
        choice = st.sidebar.multiselect('select columns', dataframe.columns.tolist())
        transformed_df = pp.manage_columns(dataframe, columns=choice, select_columns=True)

    if drop_dupl_col:
        choice = st.sidebar.selectbox('drop duplicates across', ['rows', 'columns', 'both'])
        with st_stdout("info"):
            st.dataframe(pp.manage_columns(transformed_df, drop_duplicates=choice))

