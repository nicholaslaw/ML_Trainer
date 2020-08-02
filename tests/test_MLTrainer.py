import pytest
import numpy as np
import pandas as pd
from MLTrainer import MLTrainer

def test_MLTrainer():
    np.random.seed(100)
    X_train = np.random.normal(size=(50, 2))
    y_train = np.random.randint(2, size=50)
    X_test = np.random.normal(size=(10, 2))
    models = MLTrainer(ensemble=True, linear=True, naive_bayes=True, neighbors=True, svm=True, decision_tree=True, seed=100)
    model_names = set(models.model_keys.keys())
    models.fit(X=X_train, Y=y_train, n_folds=5, scoring="accuracy", n_jobs=-1, gridsearchcv=False, param_grids={}, greater_is_better=True)
    predictions = models.predict(X_test)
    pred_proba = models.predict_proba(X_test)

    # Test for existence of models
    assert len(models.models) > 0
    
    # Test that cv_scores is a Pandas DataFrame
    assert isinstance(models.cv_scores, pd.DataFrame)

    # Test that models have been trained
    assert models.fitted == True

    # Test that predictions are contained within a dictionary
    assert isinstance(predictions, dict)
    assert isinstance(pred_proba, dict)

    # Test that initiated models are generating outputs
    assert len(model_names & set(predictions.keys())) == len(model_names)
    assert len(model_names & set(pred_proba.keys())) == len(model_names)