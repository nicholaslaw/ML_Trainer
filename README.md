# Sklearn's Auto Model Trainer

# Installation
1. docker
```
docker-compose up
```

2. pip
```
pip install . # install package
pip install -r requirements.txt # install jupyter and pytest for testing
```

3. shell script
```
./setup.sh docker # this is essentially step (1)

./setup.sh pip # use this instead if want step (2)
```

# Getting Started
```
from MLTrainer import MLTrainer

models = MLTrainer(ensemble=True, linear=True, naive_bayes=True, neighbors=True, svm=True, decision_tree=True, seed=100)
models.fit(X=X_train, Y=y_train, n_folds=5, scoring="accuracy", n_jobs=-1, gridsearchcv=False, param_grids={}, greater_is_better=True)
predictions = models.predict(X_test)
pred_proba = models.predict_proba(X_test)
```

# Jupyter Notebook Server
To set up a notebook server, follow step 1 or 3 of Installation and assuming default settings are applied, head to http://localhost:8889/tree to view existing or create new notebooks to perform experiments with the module. Password would be password.

# Testing Your Alterations
```
pytest tests/test_MLTrainer.py
```
or
```
./test.sh
```