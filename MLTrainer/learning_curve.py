import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer
from typing import Union

class LearningCurve:
    def __init__(self, batch_size: int=100, cv: int=5, seed: int=42, shuffle: bool=False):
        """
        PARAMS
        ==========
        batch_size: int
            number of samples in training batch
        cv: int
            number of cross validation folds
        seed: int
            random_state for reproducibility
        shuffle: bool
            True if want shuffe for cross validation

        RETURNS
        ==========
        """
        self.batch_size = batch_size
        self.cv = cv
        self.seed = seed
        self.shuffle = shuffle

    def compute_scores(self, model, train_X: Union[tuple, list, np.ndarray], train_Y: Union[tuple, list, np.ndarray], test_X: Union[tuple, list, np.ndarray], test_Y: Union[tuple, list, np.ndarray], scoring: str="accuracy"):
        """
        PARAMS
        ==========
        model: sklearn model
            untrained model
        train_X: numpy array
            shape (num_samples, num_features)
        train_Y: numpy array
            shape (num_samples, )
        test_X: numpy array
            shape (num_samples, num_features)
        test_Y: numpy array
            shape (num_samples,)
        scoring: str
            scoring string based on https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        RETURNS
        ==========
        Pandas Dataframe containing cv and test scores for each training batch size
        """
        skf = StratifiedKFold(n_splits=self.cv, random_state=self.seed, shuffle=self.shuffle)
        skf_cv = StratifiedKFold(n_splits=len(train_X) // self.batch_size, random_state=self.seed, shuffle=self.shuffle)
        cv_scores = []
        test_scores = []
        scoring_func = get_scorer(scoring)._score_func
        batch_idx = [0]

        for _, test_index in skf_cv.split(train_X, train_Y):
            test_index = list(test_index)
            sub_train_X = train_X[test_index]
            sub_train_Y = train_Y[test_index]
            score = np.mean(cross_val_score(model, sub_train_X, sub_train_Y, scoring=scoring, cv=skf))
            cv_scores.append(score)

            model_clone = sklearn.base.clone(model)
            model_clone.fit(sub_train_X, sub_train_Y)
            predictions = model_clone.predict(test_X)
            test_scores.append(scoring_func(test_Y, predictions))

            next_ele = batch_idx[-1] + len(test_index)
            batch_idx.append(next_ele)

        return pd.DataFrame({"Sample Size": batch_idx[1:],"train_"+scoring: cv_scores, "test_"+scoring: test_scores})

    def plot(self, df: pd.DataFrame, savefig: str=None):
        """
        PARAMS
        ==========
        df: pd.DataFrame
            output df from self.compute_scores()
        savefig: str
            specify file path to save plot

        RETURNS
        ==========
        plot learning curve
        """
        train_col = [i for i in df.columns if "train_" in i][0]
        test_col = [i for i in df.columns if "test_" in i][0]
        scoring = train_col.split("_")[-1]
        plt.plot("Sample Size", train_col, data=df, markersize=4, color='blue', linewidth=2)
        plt.plot("Sample Size", test_col, data=df, markersize=4, color='red', linewidth=2)
        plt.xlabel("Sample Size")
        plt.ylabel(scoring.capitalize())
        plt.title("Learning Curve")
        plt.legend()
        plt.show()
        
        if savefig:
            plt.savefig(savefig)