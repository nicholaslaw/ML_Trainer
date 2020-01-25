from setuptools import setup

setup(
   name='MLTrainer',
   version='1.0',
   description='A module to for automated training of traditional machine learning models',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['MLTrainer'],  #same as name
   install_requires=[
       "numpy==1.17.2",
       "scikit-learn==0.20.3",
       "pandas==0.25.1",
       "joblib==0.13.0"
   ], #external packages as dependencies
)