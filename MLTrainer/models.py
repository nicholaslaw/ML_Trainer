from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection, metrics
import numpy as np, pandas as pd, logging, os, joblib
from .model_params import classf_grids
import warnings
from typing import Union

class MLTrainer:
    def __init__(self, ensemble: bool=True, linear: bool=True, naive_bayes: bool=True, neighbors: bool=True, svm: bool=True, decision_tree: bool=True, seed: int=100) -> None:
        """
        PARAMS
        ==========
        ensemble: bool
            True if want ensemble models
        linear: bool
            True if want linear models
        naive_bayes: bool
            True if want naive bayes models
        neighbors: bool
            True if want neighbors models
        svm: bool
            True if want svm models
        decision tree: bool
            True if want decision tree models
        NOTE: Need fix naive bayes and folder names
        """
        self.models = [] # list containing names of models, i.e. strings
        self.n_classes = None # Number of classes
        self.fitted = False
        self.ensemble = ensemble
        self.linear = linear
        self.naive_bayes = naive_bayes
        self.neighbors = neighbors
        self.svm = svm
        self.decision_trees = decision_tree
        self.seed = seed
        self.cv_scores = dict()
        self.model_keys = dict()
        self.idx_label_dic = dict()
        self.init_all_models()
        
    def init_ensemble(self) -> None:
        all_models = [ensemble.AdaBoostClassifier(), ensemble.BaggingClassifier(), ensemble.ExtraTreesClassifier(),
                        ensemble.GradientBoostingClassifier(), ensemble.RandomForestClassifier()]
        self.models.extend(all_models)
        models = ["adaboost", "bagging", "extratrees", "gradientboosting", 'randomforest']
        for mod in models:
            self.model_keys[mod] = "ensemble"
        
    def init_linear(self) -> None:
        all_models = [linear_model.LogisticRegression()]
        self.models.extend(all_models)
        models = ["logreg"]
        for mod in models:
            self.model_keys[mod] = "linear"

    def init_naive_bayes(self) -> None:
        """
        MultinomialNB works with occurrence counts
        BernoulliNB is designed for binary/boolean features
        """
        all_models = [naive_bayes.BernoulliNB(), naive_bayes.GaussianNB(), naive_bayes.MultinomialNB(), naive_bayes.ComplementNB()]
        self.models.extend(all_models)
        models = ["bernoulli", "gaussian", "multinomial", "complement"]
        for mod in models:
            self.model_keys[mod] = "nb"

    def init_neighbors(self) -> None:
        all_models = [neighbors.KNeighborsClassifier()]
        self.models.extend(all_models)
        models = ["knn"]
        for mod in models:
            self.model_keys[mod] = "neighbors"

    def init_svm(self) -> None:
        all_models = [svm.NuSVC(probability=True), svm.SVC(probability=True)]
        self.models.extend(all_models)
        models = ["nu", "svc"]
        for mod in models:
            self.model_keys[mod] = "svm"

    def init_decision_tree(self) -> None:
        all_models = [tree.DecisionTreeClassifier(), tree.ExtraTreeClassifier()]
        self.models.extend(all_models)
        models = ["decision", "extra"]
        for mod in models:
            self.model_keys[mod] = "tree"

    def init_all_models(self) -> None:
        if self.ensemble:
            self.init_ensemble()
        if self.linear:
            self.init_linear()
        if self.naive_bayes:
            self.init_naive_bayes()
        if self.neighbors:
            self.init_neighbors()
        if self.svm:
            self.init_svm()
        if self.decision_trees:
            self.init_decision_tree()
        if len(self.models) == 0:
            raise Exception("No Models Selected, Look at the Parameters of ___init__")

    def fit(self, X: Union[tuple, list, np.ndarray], Y: Union[tuple, list, np.ndarray], n_folds: int=5, scoring: str="accuracy", n_jobs: int=-1, gridsearchcv: bool=False, param_grids: dict={}, greater_is_better: bool=True):
        """
        PARAMS
        ==========
        X: numpy array
            shape is (n_samples, n_features)
        Y: numpy array
            shape is (n_samples,)
        n_folds: int
            number of cross validation folds
        njobs: int
            sklearn parallel
        scoring: str
            string indicating scoring metric, reference can be found at https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        gridsearchcv: bool
            True if want parameter search with gridsearch
        param_grids: nested dictionary
            contains several parameter grids
        greater_is_better: bool
            True if the evaluation metric is better when it is greater, the results dataframe will be sorted with ascending = not greater_is_better
        """
        self.n_classes = len(np.unique(Y))
        cv_metric = "mean_cv_"+scoring
        self.cv_scores = {"model": [], "parameters": [], cv_metric: [], "remarks": []}

        if gridsearchcv:
            param_grids = classf_grids

        counter = 0
        for model_name, model in zip(list(self.model_keys.keys()),self.models):

            if gridsearchcv:
                mod = model_selection.GridSearchCV(model, param_grids[self.model_keys[model_name]][model_name], n_jobs=n_jobs)
            else:
                mod = model
                if hasattr(mod, "n_jobs"):
                    mod.n_jobs = n_jobs
                if hasattr(mod, "random_state"):
                    mod.random_state = self.seed
                mod.set_params(**param_grids.get(model_name, dict()))

            params = None
            score = None
            remark = ""

            try:
                if gridsearchcv:
                    mod.fit(X, Y)
                    score = mod.best_score_
                    mod = mod.best_estimator_
                else:
                    score = np.mean(model_selection.cross_val_score(mod, X, Y, cv=n_folds, scoring=scoring))
                    mod.fit(X, Y)
                params = mod.get_params()

            except Exception as e:
                remark = e

            self.models[counter] = mod
            counter += 1
            self.cv_scores["model"].append(model_name)
            self.cv_scores["parameters"].append(params)
            self.cv_scores["remarks"].append(remark)
            self.cv_scores[cv_metric].append(score)
        
        self.cv_scores = pd.DataFrame(self.cv_scores)
        self.fitted = True

        return self

    def predict(self, X: Union[tuple, list, np.ndarray]) -> dict:
        """
        PARAMS
        ==========
        X: numpy array
            shape is (n_samples, n_features)

        RETURNS
        ==========
        test_Y: numpy array
            shape is (n_samples,)
        """
        assert self.fitted == True, "Call .fit() method first"
        result = dict()
        model_names = list(self.model_keys.keys())

        for idx, model in enumerate(self.models):
            model_name = model_names[idx]
            try:
                predictions = model.predict(X)
            except Exception as e:
                predictions = e

            result[model_name] = predictions

        return result

    def predict_proba(self, X: Union[tuple, list, np.ndarray]) -> dict:
        """
        PARAMS
        ==========
        X: numpy array
            shape is (n_samples, n_features)

        RETURNS
        ==========
        test_Y: numpy array
            shape is (n_samples,)
        """
        assert self.fitted == True, "Call .fit() method first"
        result = dict()
        model_names = list(self.model_keys.keys())

        for idx, model in enumerate(self.models):
            model_name = model_names[idx]
            try:
                proba = model.predict_proba(X)
            except Exception as e:
                proba = e

            result[model_name] = proba

        return result

    def evaluate(self, test_X: Union[tuple, list, np.ndarray], test_Y: Union[tuple, list, np.ndarray], idx_label_dic: dict=None, class_report: str="classf_report.csv", con_mat: str="confusion_matrix.csv", pred_proba: str="predictions_proba.csv") -> None:
        """
        PARAMS
        ==========
        test_X: numpy array
            shape is (n_samples, n_features), test features
        test_Y: numpy array
            shape is (n_samples, 1), test labels
        idx_label_dic: dictionary
            keys are indices, values are string labels
        class_report: str
            file path to save classification report
        con_mat: str
            file path to save confusion matrix
        pred_proba: str
            file path to save csv containing prediction probabilities

        RETURNS
        ==========
        Saves classification report, confusion matrix and label probabilities in CSV
        """
        assert self.fitted == True, "Call .fit() method first"
        if idx_label_dic is None:
            idx_label_dic = {idx: str(idx) for idx in range(self.n_classes)}
        self.idx_label_dic = idx_label_dic
        del idx_label_dic
        for model_name, model in zip(list(self.model_keys.keys()) ,self.models):
            folder = "./" + model_name + "/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.evaluate_model(model, test_X, test_Y, folder, class_report=class_report, con_mat=con_mat, pred_proba=pred_proba)

    def evaluate_model(self, model, test_X: Union[tuple, list, np.ndarray], test_Y: Union[tuple, list, np.ndarray], folder: str="", class_report: str="classf_report.csv", con_mat: str="confusion_matrix.csv", pred_proba: str="predictions_proba.csv") -> None:
        """
        PARAMS
        ==========
        model: Sklearn model object
        test_X: numpy array
            shape is (n_samples, n_features), test features
        test_Y: numpy array
            shape is (n_samples, 1), test labels
        folder: string
            path to folder where all files are saved in
        class_report: string
            path to save classification report in csv
        confusion_mat: string
            path to save confusion matrix
        pred_proba: string
            path to save predicted probabilities

        RETURNS
        ==========
        Saves classification report, confusion matrix and label probabilities in CSV
        """
        try:
            predictions = model.predict(test_X)
            predictions_proba = model.predict_proba(test_X)
        except:
            return
        else:
            self.save_classf_report(metrics.classification_report(test_Y, predictions, labels=list(self.idx_label_dic.keys())), folder+class_report) # Save sklearn classification report in csv
            self.save_conf_mat(test_Y, predictions, folder+con_mat)
            self.save_label_proba(predictions_proba, folder+pred_proba)


    def save_classf_report(self, report, file_path: str):
        """
        PARAMS
        ==========
        report: sklearn classification report
        file_path: string
            path to save classification report as csv

        RETURNS
        ==========
        Saves classification report in CSV
        """
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-4]:
            row_data = line.split()
            if len(row_data) != 0:
                row = {}
                row['precision'] = float(row_data[-4])
                row['recall'] = float(row_data[-3])
                row['f1_score'] = float(row_data[-2])
                row['support'] = float(row_data[-1])
                row['class'] = self.idx_label_dic[int(row_data[0])]
                report_data.append(row)
        df = pd.DataFrame.from_dict(report_data)
        df.to_csv(file_path, index=False)

    def save_conf_mat(self, test_Y: Union[tuple, list, np.ndarray], predictions: Union[tuple, list, np.ndarray], file_path: str):
        """
        PARAMS
        ==========
        test_Y: numpy array
            shape is (n_samples, 1), true labels
        predictions: numpy array
            shape is (n_samples, 1), predicted labels
        file_path: string
            path to save confusion matrix
        
        RETURNS
        ==========
        Saves confusion matrix in CSV
        """
        confusion_mat = metrics.confusion_matrix(test_Y, predictions, labels=list(self.idx_label_dic.keys()))
        total_row = confusion_mat.sum(axis=0)
        total_col = [np.nan] + list(confusion_mat.sum(axis=1)) + [sum(total_row)]
        confusion_mat_df = pd.DataFrame({})
        confusion_mat_df["Predicted"] = ["True"] + list(self.idx_label_dic.values()) + ["All"]
        for idx, label in self.idx_label_dic.items():
            temp = [np.nan] + list(confusion_mat[:, idx]) + [total_row[idx]]
            confusion_mat_df[label] = temp

        confusion_mat_df["All"] = total_col
        confusion_mat_df.to_csv(file_path, index=False)

    def save_label_proba(self, pred_proba: np.ndarray, file_path: str):
        """
        PARAMS
        ==========
        pred_proba: numpy array
            shape is (n_samples, 1), predicted probabilities
        file_path: string
            file path to save label probabilities in CSV

        RETURNS
        ==========
        Saves label probabilities in CSV
        """
        proba_df = pd.DataFrame({})
        for idx, label in self.idx_label_dic.items():
            proba_df[label] = pred_proba[:, idx]
        proba_df.to_csv(file_path, index=False)