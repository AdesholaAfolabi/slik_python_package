import docs
import paths
import streamlit as st

from PIL import Image
from utils import (
    _divider, app_meta, app_section_button,
    display_app_header, display_sidebar_url_log
)


# Add application meta data
app_meta()

# Side Bar
# Links to Getting Started and Exploring
# Slik Wrangler would be resolved upon deployment
with st.sidebar:
    st.markdown(
        body="<b style='font-size: 18px;font-weight: 900;text-aligh: center'>The Slik-Wrangler Web Application!!!</b>",
        unsafe_allow_html=True
    )
    st.image(Image.open(paths.DATA_LOAD_IMAGE_PATH))
    _divider()
    display_sidebar_url_log(
        _docs=[
            "Haven't used Slik-wrangler and want to see it in action?",
            "So you have used Slik-wrangler, but want to get deep into the Nitty-gritty",
            "Want to contribute to making Slik-wrangler better?",
            "Take a look at the developers working on Slik-wrangler"
        ],
        _urls=[
            "[Get Started with Slik-wrangler 🚀](http://localhost:8502)",
            "[Review the Documentation 🗒](https://sensei-akin.github.io/slik_python_package/html/index.html)",
            "[Contribute to the package 💻️](https://github.com/AdesholaAfolabi/slik_python_package)",
            "[The Slik Team ❤](http://localhost:8503)"
        ]
    )
    st.markdown(
        body="<b style='text-align: center'>Slik-Wrangler, ©️ 2022</b>",
        unsafe_allow_html=True
    )

# Fancy Header
# Slik Wrangler default header
display_app_header(
    main_txt='Slik-Wrangler Web Application',
    sub_txt='Clean 🧼, describe 🗣️, visualise 📊 and model 🚢'
)
_divider()

# Introducing Slik Wrangler
# Short Introduction to Slik Wrangler
st.write(docs.ABOUT_SLIK_WRANGLER_1)

video_tab, preprocess_tab, contact_tab = st.tabs([
    "Watch a demo",
    "Preprocess with Slik-Wrangler",
    "Contact Us"
])

with video_tab:
    st.write(docs.ABOUT_SLIK_WRANGLER_2)
    st.video(open("../data/video_data/Intro_video.mp4", 'rb').read())

with preprocess_tab:
    st.write(docs.ABOUT_SLIK_WRANGLER_3)

with contact_tab:
    st.write(docs.ABOUT_SLIK_WRANGLER_4)


# You can't control your actions if you don't have control
# over your thoughts. You can't control your thought if you
# don't know you have that option.

