import docs
import streamlit as st

from utils import (
    app_meta, app_section_button, display_app_header, divider
)


# Add application meta data
app_meta()

# Fancy Header
# Slik Wrangler default header
display_app_header(
    main_txt='Slik-Wrangler Web Application',
    sub_txt='Clean, describe, visualise and model'
)
divider()

# Fancy Navigation Bar
# Links to Getting Started and Exploring
# Slik Wrangler would be resolved upon deployment
app_section_button(
    "[Getting Started 🚀](http://localhost:8502)",
    "[The Slik Team ❤](http://localhost:8503)",
    "[Contribute 💻️](https://github.com/AdesholaAfolabi/slik_python_package)",
    "[Documentation 🗒](https://sensei-akin.github.io/slik_python_package/html/index.html)"
)
divider()

# Introducing Slik Wrangler
# Short Introduction to Slik Wrangler
st.write(docs.ABOUT_SLIK_WRANGLER_1)
divider()
st.video(open("../data/video_data/Intro_video.mp4", 'rb').read())
divider()
st.write(docs.ABOUT_SLIK_WRANGLER_2)
