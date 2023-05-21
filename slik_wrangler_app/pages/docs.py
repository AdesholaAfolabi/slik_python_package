import streamlit as st

from PIL import Image
from paths import randomly_generate_plain_image_path


LINK_TO_VIDEO = "https://static.streamlit.io/examples/star.mp4"

ABOUT_SLIK_WRANGLER_1 = """
slik-wrangler is a **data to modeling tool** that helps data scientists navigate the issues of basic **data wrangling and preprocessing steps**. The idea behind slik-wrangler is to jump-start supervised learning projects. 

Data scientists struggle to prepare their data for building machine learning models and all machine learning projects require data wrangling, data preprocessing, feature engineering which takes about 80% of the model building process.
"""

ABOUT_SLIK_WRANGLER_2 = """
slik-wrangler has several tools that make it easy to load data of any format, clean and inspect your data. It offers a quick way to pre-process data and perform feature engineering. Building machine learning models is an inherently iterative task and data scientists face challenges of reproducing the models and productionalizing model pipelines.
"""

ABOUT_SLIK_WRANGLER_3 = """
With slik-wrangler, Data scientists can build model pipelines. slik-wrangler provides explainability in the pipeline process in the form of DAG showing each step in the build process. With every build process/experiment, slik-wrangler logs the metadata for each run.

### Example

A minimum example of using slik-wrangler for preprocessing is:

```python
from slik_wrangler import preprocessing as pp

from sklearn.model_selection import train_test_split

from sklearn.datasets import titanic

X, y = titanic(return_X_y=True)

pp.preprocess(
    data=X, target_column='Survived', train=True, 
    verbose=False, project_path='./Titanic', logging='display'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lr = LogisticRegression.fit(X_train, y_train)

print("Accuracy score", lr.score(X_test, y_test))
```
"""

ABOUT_SLIK_WRANGLER_4 = """
You can reach us by mail at slik-wrangler@gmail.com

Other ways you can know about what we're doing is by following us at our social accounts at

1. Twitter 👉️ [@Slik-Wrangler](https://twitter.com/slikwrangler?t=4tucupjpYg1QWW0oHGQ_hw&s=09)
2. LinkedIn 👉️ [@@Slik-Wrangler](https://www.linkedin.com/company/slik-wrangler)

Other ways to reach is through our [team](http://localhost:8503).

We are actively seeking contribution to continue improving our open source project. Any kind of help is welcome. Just a star on the project is a lot. If you would like to contribute as a developer, you can join the project by [filling out this form](https://forms.gle/s88QBMXEzfaRB66s6) or by opening an issue. Any other kind of contribution, from docs to tests, is also welcome.

📣 Please fill out our [1 min survey](👯‍♀️) so that we can learn what do you think about Slik-wrangler, how you are using it, and what improvements we should make. Thank you! 👯‍


"""

FILE_UPLOADER_TEXT = """
Enter the path of a data file with any of this extensions

["csv", "xlsx", "xls", "parquet", "json"]

Ensure the data file(s) size doesn't exceed 200MB
"""

DATA_LOADING_ASSESSMENT_1 = """
# Data Loading & Data Quality Assessment (DQA)

Data Quality Assessment (DQA) is the process of asserting the quality of the data (or dataset). The process of asserting data quality ensures the data is suitable for use and meets the quality required for projects or business processes.
"""

DATA_LOADING_ASSESSMENT_2 = """
Slik-wrangler handles the data quality assessment with a dedicated module called dqa which contains several functions for checking the data quality. One of the functions which would be often used is the data_cleanness_assessment which shows an overview of how clean the data is:

```python
from slik_wrangler.dqa import data_cleanness_assessment

data_cleanness_assessment(dataset)
```

This is all you need to get an overview of how clean your dataset is and if there are any issues to be addressed. Slik-dqa also provides independent functions for checking specific issues with the dataset, functions like: missing_value_assessment, duplicate_assessment, e.t.c.

To load you data, enter the path to where your dataset is located in you project directory, and then proceed to loading and asserting your dataset.
"""

PAGE_NOT_AVAILABLE_MESSAGE = """
<b style="text-align: center;">Hey 👋, Check back later, this section is currently undergoing construction.</b>
"""

FOLLOW_SIDEBAR_INSTRUCTION = """
<p style="padding-top: 50px; font-size: 20px; text-align: center">You are just starting out this session 🤗</p>

<p style="font-size: 20px; text-align: center">In order to properly load work with the Slik-Wrangler package on streamlit and 
start the project, you have to click the start project button on the side-bar</p>

<p style="font-size: 100px; text-align: center">👈️</p>
"""


def section_not_available(message=None, add_plain_image=False):
    if message is None:
        message = PAGE_NOT_AVAILABLE_MESSAGE

    st.markdown(message, unsafe_allow_html=True)

    if add_plain_image:
        st.image(Image.open(randomly_generate_plain_image_path()))

