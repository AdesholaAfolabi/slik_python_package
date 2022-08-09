#!/usr/bin/python3
import sys
sys.path.insert(0, './')
from slik_wrangler import loadfile as lf
from slik_wrangler import preprocessing as pp
from slik_wrangler import dqa 
import streamlit as st
from datetime import datetime
import docs,os,paths
from PIL import Image
import tabula as tb
import plotly.express as px
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import pickle
import pandas as pd
import seaborn as sns
# from slik_wrangler import preprocessing as pp

# from slik_wrangler.loadfile import _read_file
matplotlib.use('Agg')

# dateparse = lambda x: datetime.strptime(x, '%d.%m.%y')

def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    fig = plt.figure(figsize=(20,10))
    plt.imshow(mywordcloud,interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

from contextlib import contextmanager
from io import StringIO

from streamlit.scriptrunner import script_run_context
from threading import current_thread
import streamlit as st
import sys


@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), script_run_context.SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield



    
def sub_text(text):
    '''
    A function to neatly display text in app.
    Parameters
    ----------
    text : Just plain text.
    Returns
    -------
    Text defined by html5 code below.
    '''
    
    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """
    
    st.markdown(html_temp, unsafe_allow_html = True)

    
@st.cache(allow_output_mutation=True)
def load_data(file):
    dataframe = lf._read_file(file)
    return dataframe



def main():
    st.set_page_config(
    page_title="Slik-wrangler UI",
    page_icon='./slik_wrangler_app/data/image_data/slik_wrangler_logo.jpeg',#"✅",
    layout="wide",
)
    st.title("Slik-wrangler UI")
    st.image('./slik_wrangler_app/data/image_data/slik_wrangler_logo.jpeg', width=600)
    url = 'https://sensei-akin.github.io/slik_python_package/html/index.html'
    st.write("""This library is under active development""" )
    st.markdown('📖 [Documentation](https://sensei-akin.github.io/slik_python_package/html/index.html)')
           

    menu = ['Home','About']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':

        file = st.file_uploader('Upload a file', type = ['pdf','csv','parquet'])
        if file is not None:
            dataframe = pd.read_csv(file)
            # st.text('Columns present in the dataframe')
            # st.write(dataframe.columns.tolist())
            st.write("## Dataframe:", dataframe.head())

            dqa_fn = st.sidebar.checkbox('Data quality assessment')
            bin_age_fn = st.sidebar.checkbox('Convert age columns to bins')
            change_case_fn = st.sidebar.checkbox('Change case')
            check_nan_fn = st.sidebar.checkbox('Explore missing values')
            manage_col_fn = st.sidebar.checkbox('Handle/Manage columns')

            transformed_df = dataframe.copy()

            if dqa_fn:
                choice = st.sidebar.radio('quality assessment',['missing value','duplicate assessment'])
                with st_stdout("info"):
                    if choice == 'missing value':
                        st.write(dqa.missing_value_assessment(dataframe))
                    else:
                        print('Function currenly undergoing development')
            if bin_age_fn:
                choice = st.sidebar.selectbox('select age column', transformed_df.columns.tolist())
                add_prefix = st.sidebar.radio("add prefix",(True, False),horizontal=True)
                transformed_df = pp.bin_age(transformed_df,choice,add_prefix)

            if change_case_fn:
                cols_choice = st.sidebar.multiselect('select columns to apply case transformation',
                 transformed_df.columns.tolist())
                if cols_choice:
                    case_choice = st.sidebar.selectbox('select the type of case', ['lower','upper','capitalize'])
                    inplace = st.sidebar.radio("inplace",(True, False),horizontal=True)
                    if not inplace:
                        new_col = st.text_input(label='Name of the new transformed column')
                        transformed_df[new_col] = pp.change_case(transformed_df,cols_choice,case_choice,inplace)
                    else:
                        transformed_df = pp.change_case(transformed_df,cols_choice,case_choice,inplace)

            if check_nan_fn:
                col1, col2 = st.columns(2)
                # with col1:
                view_toggle = st.radio('Choose view',["Plot","Dataframe"])
                with st_stdout("info"):
                    if view_toggle=='Plot':
                        # nan_values = df.set_index('features')['missing_percent']
                        pp.check_nan(transformed_df,plot=True,streamlit=True)
                        # plot_nan(df)
                    else:
                        st.dataframe(pp.check_nan(transformed_df,display_inline=True))

            if manage_col_fn:
                manage_sel = st.sidebar.radio(
                "Manage columns",
                ("","select columns", "drop columns")
                )
                
                drop_dupl_col = st.sidebar.checkbox("drop duplicates")
                
                
                if manage_sel=='drop columns':
                    drop_columns = True
                    choice = st.sidebar.multiselect('drop columns',dataframe.columns.tolist())
                    transformed_df = pp.manage_columns(transformed_df,columns=choice,drop_columns=drop_columns)
                elif manage_sel=='':
                    pass
                else:
                    choice = st.sidebar.multiselect('select columns',dataframe.columns.tolist())
                    transformed_df = pp.manage_columns(transformed_df,columns=choice,select_columns=True)

                if drop_dupl_col:
                    choice = st.sidebar.selectbox('drop duplicates across', ['rows','columns','both'])
                    with st_stdout("info"):
                        transformed_df = pp.manage_columns(transformed_df,drop_duplicates=choice)
                    
            preview_button = st.checkbox(label='preview transformed data')
            if preview_button:
                st.write("## Transformed Dataframe:")
                st.dataframe(transformed_df)

            
    else:
        st.subheader("About")
        st.write(docs.ABOUT_SLIK_WRANGLER_1)
        # call back function -> runs BEFORE the rest of the app
        def reset_button():
            st.session_state["p"] = False
            return 

        #button to control reset
        reset=st.button('Reset', on_click=reset_button)

        

        #checkbox you want the button to un-check
        check_box = st.checkbox("p", key='p')
        if check_box:
            with st_stdout("info"):
                print('quick look')
        #write out the current state to see that our flow works
        st.write(st.session_state)
    

if __name__ == '__main__': 

    main()