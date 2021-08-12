# slik_python_package
This README documents the steps that are necessary to get the slik package up and running.
 <img  src="https://github.com/Sensei-akin/slik_python_package/blob/master/docs/_images/slik.png" 
### What is this repository for? ###

* Quick summary: The application employs the modular style of putting the applications together. In total, there are four modules which takes care of reading any type of file, data preprocessing (Nan, outliers, etc), and other preprocessing steps such as One Hot Encoding, Scaling, Normalization, PCA, etc. There is a general module (which is the general_utils module) that contains a list of global attributes and data used throughout the project. 
* Version: 1.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up: ensure you have python 3 up and running
* Configuration: ensure all modules are imported properly. They all depend on each other
* Dependencies: python 3, pandas, scikit-learn, sklearn pre-processing
* Database configuration: no required configuration
* How to run tests: no tests files used yet. Version 2 will come with test cases
* Deployment instructions: To use this package, use the Savepipeline method (which requires a file path and input columns as input parameters) in save_object module and call the compile_functions. The final output is a csv file and a pickle pipeline object.

"README.md" 27L, 1205C
* Repo owner: afolabimkay@gmail.com
