from setuptools import setup

setup(
   name='MLTrainer',
   version='1.0',
   description='A module to for automated training of traditional machine learning models',
   author='Nicholas Law',
   author_email='nicholas_law_91@hotmail.com',
   packages=['MLTrainer'],
   install_requires=[
       "numpy==1.18.1",
       "scikit-learn==0.22.2.post1",
       "pandas==1.0.1",
       "joblib==0.14.1",
       "matplotlib==3.3.0",
       "xgboost==1.3.1"
   ], #external packages as dependencies
)