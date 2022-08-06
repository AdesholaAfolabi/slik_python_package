# Structuring the Slik-Wrangler Web Application

import docs
import paths
import streamlit as st

from PIL import Image
from utils import page_decor

from slik_wrangler.loadfile import read_file

from slik_wrangler.dqa import data_cleanness_assessment

@page_decor
def intro_page():
    with st.container():
        st.write(docs.ABOUT_SLIK_WRANGLER_1)
        st.video(open(paths.INTRO_VIDEO_PATH, 'rb').read())
        st.write(docs.ABOUT_SLIK_WRANGLER_2)


@page_decor
def data_page():
    st.write(docs.DATA_LOADING_ASSESSMENT_1)
    st.image(
        Image.open(paths.DATA_LOAD_IMAGE_PATH),
        caption='Slik-Wrangler Wallpaper'
    )
    st.write(docs.DATA_LOADING_ASSESSMENT_2)

    file = st.text_input(
        label="Data File",
        help=docs.FILE_UPLOADER_TEXT
    )

    with st.sidebar:
        if file is not None:
            pass

    data_loading_tab, quality_assessment_tab, nan_count_tab = st.tabs(
        [
            "Data Loading",
            "Data Assessment",
            "Missing Values Chart"
        ]
    )

    with data_loading_tab:
        st.write("### Read & view file")
        if st.button("Read File"):
            if file is not None:
                dataframe = read_file(file)
                st.write(dataframe)

    with quality_assessment_tab:
        st.write("### Assert File")
        if st.button("Assert Data File"):
            if file is not None:
                dataframe = read_file(file)
                st.write(
                    *data_cleanness_assessment(
                        dataframe, return_as_str=True
                    )
                )

    with nan_count_tab:
        # To be computed
        pass


page_names_to_funcs = {
    "Main Page": intro_page,
    "Data Loading & Assessment": data_page
}

selected_page = st.sidebar.selectbox("Open page...", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
