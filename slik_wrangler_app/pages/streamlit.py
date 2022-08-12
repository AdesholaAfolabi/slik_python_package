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


@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



from contextlib import contextmanager
from io import StringIO

from streamlit.scriptrunner import script_run_context
from threading import current_thread
import streamlit as st
import sys


@contextmanager
def st_redirect(src, dst,how=None):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)
    s = []

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), script_run_context.SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
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
        # st_redirect.value = new_write.value


@contextmanager
def st_stdout(dst,how=None):
    with st_redirect(sys.stdout, dst,how):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


                

# @st.cache(allow_output_mutation=True)
def plot_corr(dataframe):
    fig = px.imshow(dataframe.corr(method='pearson'), 
                title='Correlation Plot', height=500, width=700)
    fig.update_layout(uniformtext_minsize=6, uniformtext_mode='hide',yaxis=dict(showgrid=False))
               
    st.write(fig)
    
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

        file = st.file_uploader('Upload a file', type = ['pdf','csv','parquet','xlsx','json'])
        if file is not None:
            dataframe = load_data(file)
            # st.text('Columns present in the dataframe')
            # st.write(dataframe.columns.tolist())
            st.write("## Dataframe:", dataframe.head())
            # st.text('')
            project_name = st.text_input('project name',help="""Specify a project name where the 
            preprocessed data and other metadata will be stored""")
            transformed_df = dataframe.copy()
            col_list = transformed_df.columns.tolist()
            schema_fn = st.checkbox('view data schema')
            if schema_fn:
                save=False
                with st_stdout("info",how='json'):
                    pp.create_schema_file(transformed_df,save=save)
                    save_schema_fn = st.checkbox('save schema')
                    if save_schema_fn:
                        output_path =  os.path.join(os.getcwd(),project_name)
                        
                        if project_name:
                            save=True
                            pp.create_schema_file(transformed_df,project_path= output_path,save=save)
                        else:
                            st.warning('Project name is yet to be defined')

            st.sidebar.subheader('Quick Preprocessing')
            datefield_fn = st.sidebar.checkbox('Assert Datefield')
            bin_age_fn = st.sidebar.checkbox('Bin Age columns')
            change_case_fn = st.sidebar.checkbox('Change case')
            dqa_fn = st.sidebar.checkbox('Data quality assessment')
            dui_fn = st.sidebar.checkbox('Drop uninformative fields')
            check_nan_fn = st.sidebar.checkbox('Explore missing values')
            manage_col_fn = st.sidebar.checkbox('Handle/Manage columns')
            featurize_dt_fn = st.sidebar.checkbox('Featurize Datatime columns')
            get_attr_fn = st.sidebar.checkbox('Get Attributes')


            
            options = bin_age_fn,change_case_fn,manage_col_fn,dui_fn,featurize_dt_fn
            

            if dqa_fn:
                choice = st.sidebar.radio('quality assessment',['missing value','duplicate assessment'])
                with st_stdout("info"):
                    if choice == 'missing value':
                        st.write(dqa.missing_value_assessment(dataframe))
                    else:
                        print('Function currenly undergoing development')

            if dui_fn:
                with st_stdout("info"):
                    col_list_ap = col_list.append(None)
                    index = len(col_list) -1
                    choice = st.sidebar.selectbox('exclude columns', col_list,key='dui',help="""
                    columns to be excluded from being dropped""",index=index)
                    pp.drop_uninformative_fields(transformed_df,exclude=choice)

            if datefield_fn:
                choice = st.sidebar.selectbox('select column', col_list,key='date')
                placeholder = st.empty()
                with placeholder.container():
                    st.markdown(f'###### Asserting {choice} is a date field:')
                    st.write(pp.check_datefield(transformed_df,choice))

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
                        df = pp.check_nan(transformed_df,display_inline=True)
                        df = df[df.missing_counts > 0]
                        st.dataframe(df)

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
                    choice = st.sidebar.multiselect('select columns',transformed_df.columns.tolist())
                    transformed_df = pp.manage_columns(transformed_df,columns=choice,select_columns=True)

                if drop_dupl_col:
                    choice = st.sidebar.selectbox('drop duplicates across', ['rows','columns','both'])
                    with st_stdout("info"):
                        transformed_df = pp.manage_columns(transformed_df,drop_duplicates=choice)

            
            if featurize_dt_fn:
                choice = st.sidebar.selectbox('select date/datetime column', transformed_df.columns.tolist())
                if choice: 
                    date_feat = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear','Week',\
                                'Is_month_end', 'Is_month_start', 'Is_quarter_end','Hour','Minute',\
                                'Is_quarter_start', 'Is_year_end', 'Is_year_start','Date']
                    feat_choice = st.sidebar.multiselect('select a date feature',date_feat)
                    drop = st.sidebar.radio("drop original datetime column",(True, False),horizontal=True)
                    with st_stdout("info"):
                        transformed_df = pp.featurize_datetime(transformed_df,choice,feat_choice, drop)

            if get_attr_fn:
                #choice = st.sidebar.selectbox('select target column', transformed_df.columns.tolist())
                col_list_ap = col_list.append(None)
                index = len(col_list) -1
                choice = st.sidebar.selectbox('exclude target column', col_list,key='get_attr',help="""
                target column to be excluded""",index=index)
                with st_stdout("info"):
                    st.write(pp.get_attributes(transformed_df, choice))

            
            if any((options)):
                preview_button = st.checkbox(label='preview transformed data')
                if preview_button:
                    st.write("## Transformed Dataframe:")
                    st.dataframe(transformed_df)

                    
                csv = convert_df(transformed_df)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='preprocess_data.csv',
                    mime='text/csv',
                )

            st.sidebar.subheader('EDA by Visualization')
            corr_fn = st.sidebar.checkbox('Correlation Plot')
            if corr_fn:
                choice = st.multiselect('select columns',transformed_df.columns.tolist())
                if choice:
                    plot_corr(transformed_df[choice])
                else:
                    plot_corr(transformed_df)

            

            
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