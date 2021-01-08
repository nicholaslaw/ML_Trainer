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
    },
    "xgboost": {
        'min_child_weight': list(range(1,10,2)),
        'gamma':[i/10.0 for i in range(0,5)],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree':[i/10.0 for i in range(6,10)],
        'max_depth':range(3,10,2),
        'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
        'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 1],
        'learning_rate': [0.001, 0.002, 0.005, 0.006, 0.01, 0.02, 0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2],
        'n_estimators': [50, 100, 150, 200, 250, 300,350,400,450,500, 550, 600, 650, 700, 750],
        "booster": ["gbtree", "gblinear", "dart"]
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
    "bernoulli": {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },
    "gaussian": {
        "var_smoothing": [1e-10, 1e-9, 1e-8]
    },
    "multinomial": {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    },
    "complement": {
        "alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "norm": [True, False]
    }
}

neighbors_grids = {
    "knn": {
        "n_neighbors": list(range(5, 10, 1)),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": list(range(30, 50, 5)),
        "p": [1, 2]
    }
}

svm_grids = {
    "nu": {
        "nu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "degree": [1, 2, 3, 4],
        "gamma": ["auto", "scale"],
        "coef0": [0.0, 0.1, 0.2],
        "shrinking": [False, True],
        "probability": [True],
        "tol": [1e-3, 1e-4, 1e-5]
        },
    "svc": {
        "C": [0.8, 0.9, 1.0],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "degree": [1, 2, 3, 4],
        "gamma": ["auto", "scale"],
        "coef0": [0.0, 0.1, 0.2],
        "shrinking": [False, True],
        "probability": [True],
        "tol": [1e-3, 1e-4, 1e-5]
    },
}

tree_grids = {
    "decision": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"],
        "max_features": ["auto", "sqrt", "log2", None]
    },
    "extra": {
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"]
    }
}

classf_grids = {
    "ensemble": ensemble_grids,
    "linear": linear_grids,
    "nb": nb_grids,
    "neighbors": neighbors_grids,
    "svm": svm_grids,
    "tree": tree_grids
}

def extract_test_params(grid):
    """
    PARAMS
    ==========
    grid: dict
        keys are model names given by myself and values are 
        dictionaries containing model parameters used by scikit learn models

    RETURNS
    ==========
    For each parameter list contained in a dictionary for a model, this function returns
    a dictionary containing parameters as keys and values are a list containing just 1 value each
    """
    result = dict()
    for key, val in grid.items():
        the = dict()
        for i, j in val.items():
            j = j[:1]
            the[i] = j
        result[key] = the
    return result