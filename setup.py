from setuptools import setup

setup(name='slik_wrangler',
      version='1.0.2',
      description='A data preprocessing and modeling tool',
      packages=['slik_wrangler'],
      author = 'Adeshola Afolabi/Akinwande Komolafe',
      author_email= 'afolabimkay@gmail.com',
      url = 'https://github.com/AdesholaAfolabi/slik_python_package',
      install_requires=[
            "pandas",
            "sklearn",
            "ipython", 
            "scipy", 
            "imblearn",
            "numpy", 
            "matplotlib", 
            "seaborn"
      ],
      zip_safe=False)

