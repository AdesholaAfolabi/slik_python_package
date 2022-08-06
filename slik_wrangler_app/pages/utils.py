import streamlit as st


def page_decor(page):
    """
    Decorations for each page in index

    :param page: the page to decorate
    :return: a decorate page.
    """

    def __internal():
        with st.sidebar:
            st.markdown("<h1 style='color:  #2980b9'>SLIK-WRANGLER 🚀</h1>", unsafe_allow_html=True)

        page()

    return __internal
