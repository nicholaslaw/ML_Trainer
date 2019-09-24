

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