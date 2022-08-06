LINK_TO_VIDEO = "https://static.streamlit.io/examples/star.mp4"

ABOUT_SLIK_WRANGLER_1 = """
# Slik-Wrangler Web Application

slik-wrangler is a **data to modeling tool** that helps data scientists navigate the issues of basic **data wrangling and preprocessing steps**. 

The idea behind slik-wrangler is to jump-start supervised learning projects. Data scientists struggle to prepare their data for building machine learning models and all machine learning projects require data wrangling, data preprocessing, feature engineering which takes about 80% of the model building process.

---
"""

ABOUT_SLIK_WRANGLER_2 = """
---

slik-wrangler has several tools that make it easy to load data of any format, clean and inspect your data. It offers a quick way to pre-process data and perform feature engineering. Building machine learning models is an inherently iterative task and data scientists face challenges of reproducing the models and productionalizing model pipelines.

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

This library is in very active development, so it’s not recommended for production use. Development at [github.com](github.com/AdesholaAfolabi/slik-wrangler_python_package/staging/).
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
