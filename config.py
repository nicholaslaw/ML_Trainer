

# Parameter Grids
## NOTE: Define grids for GridSearchCV
ensemble_grids = {
    "adaboost": {
        "n_estimators": list(range(50, 500, 50)),
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "algorithm": ["SAMME", "SAMME.R"]
    },
    "bagging": {
        "n_estimators": list(range(50, 500, 50)),
    },
    "extratrees":{
        "n_estimators": list(range(50, 500, 50)),
        "criterion": ["gini", "entropy"],
        "max_features": ["auto", "sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"]
    },
    "gradientboosting": {
        "loss": ["deviance", "exponential"],
        "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "n_estimators": list(range(50, 500, 50)),
        "criterion": ["friedman_mse", "mse", "mae"],
        "max_features": ["auto", "sqrt", "log2", None],
        "tol": [1e-4, 1e-3]
    },
    "randomforest": {
        "n_estimators": list(range(50, 500, 50)),
        "criterion": ["gini", "entropy"],
        "max_features": ["auto", "sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"]
    }
}

linear_grids = {
    "logreg": {
        "penalty": ["l1", "l2", "elasticnet", None],
        "tol": [1e-4, 1e-3],
        "fit_intercept": [True, False],
         "class_weight": ["balanced", None],
         "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
         "max_iter": [50, 100],
         "multi_class": ["ovr", "multinomial", "auto"],
    }
}

nb_grids = {
}