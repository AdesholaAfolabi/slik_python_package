import docs
import streamlit as st

from PIL import Image
from utils import (
    app_meta, divider, load_data, convert_df,
    display_app_header, Executor
)
from operations import (
    perform_dqa_operation, perform_bin_age_operation,
    perform_check_nan_operation, perform_date_field_operation,
    perform_dui_operation, perform_featurize_dt_operation,
    perform_get_attr_operation, perform_manage_col_operation
)


# Add application meta data
app_meta()

with st.sidebar:
    st.image(Image.open("../data/image_data/start.png"))
    st.markdown("**Step 1**")
    st.markdown("**Data Quality Assessment (DQA)**")
    start_project = st.checkbox(
        label="Start Project",
        help="Starts the Slik-Wrangler Project"
    )

if start_project:
    # Fancy Header
    # Slik Wrangler default header
    display_app_header(
        main_txt='Slik-Wrangler Web Application',
        sub_txt='Clean, describe, visualise and model'
    )
    divider()

    data_file = st.file_uploader(
        label="Data File",
        help=docs.FILE_UPLOADER_TEXT,
        type=['csv', 'xls', 'xlsx', 'parquet', 'json']
    )

    if data_file is not None:
        dataset = load_data(data_file)
        st.write(dataset)
        divider()

        with st.sidebar:
            st.markdown("Data loaded successfully ✔")
            dqa_operations = st.expander("Data Quality Assessment Operations")
            divider()
            st.image(Image.open("../data/image_data/processing.png"))
            st.markdown("**Step 2**")
            st.markdown("**Data Pre-processing**")
            st.markdown("Able to preprocess dataset ✔")
            pp_operations = st.expander("Data Pre-processing Operations")
            divider()
            st.markdown("<b style='font-size: 30px'>Control Menu 🎛️</b>", unsafe_allow_html=True)

        with dqa_operations:
            log_dqa_fn = st.checkbox("Run data quality assessment")
            dui_fn = st.checkbox('Drop uninformative fields')
            check_nan_fn = st.checkbox('Explore missing values')

        with pp_operations:
            date_field_fn = st.checkbox('Assert Datefield')
            bin_age_fn = st.checkbox('Bin Age columns')
            change_case_fn = st.checkbox('Change case')
            manage_col_fn = st.checkbox('Handle/Manage columns')
            featurize_dt_fn = st.checkbox('Featurize Datatime columns')
            get_attr_fn = st.checkbox('Get Attributes')

        options = [
            log_dqa_fn, dui_fn, check_nan_fn, date_field_fn, bin_age_fn,
            change_case_fn, manage_col_fn, featurize_dt_fn, get_attr_fn
        ]

        executor = Executor(dataset)
        executor.add_operation(log_dqa_fn, perform_dqa_operation)
        executor.add_operation(date_field_fn, perform_date_field_operation)
        executor.add_operation(bin_age_fn, perform_bin_age_operation)
        executor.add_operation(dui_fn, perform_dui_operation)
        executor.add_operation(check_nan_fn, perform_check_nan_operation)
        executor.add_operation(manage_col_fn, perform_manage_col_operation)
        executor.add_operation(featurize_dt_fn, perform_featurize_dt_operation)
        executor.add_operation(get_attr_fn, perform_get_attr_operation)
        executor.execute()

        if any(options):
            preview_button = st.sidebar.checkbox(label='preview transformed data')

            if preview_button:
                st.write("### Transformed Dataframe:")
                st.dataframe(dataset)

            csv = convert_df(dataset)

            with st.sidebar:
                st.download_button(
                    label="Download transformed data as CSV",
                    data=csv,
                    file_name='preprocess_data.csv',
                    mime='text/csv',
                )

else:
    docs.section_not_available(docs.FOLLOW_SIDEBAR_INSTRUCTION)
