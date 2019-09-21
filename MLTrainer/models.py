from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree, model_selection, metrics
import numpy as np, pandas as pd, logging, os

class MLTrainer:
    def __init__(self, ensemble=True, linear=True, naive_bayes=False, neighbors=True, svm=True, decision_tree=True, seed=100):
        """
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
        self.models = []
        self.ensemble = ensemble
        self.linear = linear
        self.naive_bayes = naive_bayes
        self.neighbors = neighbors
        self.svm = svm
        self.decision_trees = decision_tree

    def fit(self, X, Y):
        """
        X: numpy array
            shape is (n_samples, n_features)
        Y: numpy array
            shape is (n_samples, 1)
        
        Train all selected models
        """
        self.init_all_models()
        for model in self.models:
            model.fit(X, Y)

    def evaluate(self, test_X, test_Y, idx_label_dic=None):
        """
        test_X: numpy array
            shape is (n_samples, n_features), test features
        test_Y: numpy array
            shape is (n_samples, 1), test labels
        idx_label_dic: dictionary
            keys are indices, values are string labels
        """
        if idx_label_dic is None:
            idx_label_dic = {idx: str(idx) for idx in range(self.models[0].n_classes_)}
        self.idx_label_dic = idx_label_dic
        del idx_label_dic
        for idx, model in enumerate(self.models):
            self.evaluate_model(model, test_X, test_Y, "./folder_" + str(idx) + "/")

    def evaluate_model(self, model, test_X, test_Y, folder, class_report="classf_report.csv", con_mat="confusion_matrix.csv", pred_proba="predictions_proba.csv"):
        """
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

        Obtain evaluation metrics
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
        predictions = model.predict(test_X)
        predictions_proba = model.predict_proba(test_X)

        self.classf_report(metrics.classification_report(test_Y, predictions, labels=list(self.idx_label_dic.keys())), folder+class_report) # Save sklearn classification report in csv
        self.conf_mat(test_Y, predictions, folder+con_mat)
        self.save_label_proba(predictions_proba, folder+pred_proba)


    def classf_report(self, report, file_path):
        """
        report: sklearn classification report
        file_path: string
            path to save classification report as csv
        Saves classification report
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

    def conf_mat(self, test_Y, predictions, file_path):
        """
        test_Y: numpy array
            shape is (n_samples, 1), true labels
        predictions: numpy array
            shape is (n_samples, 1), predicted labels
        file_path: string
            path to save confusion matrix
        Saves confusion matrix
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

    def save_label_proba(self, pred_proba, file_path):
        """
        pred_proba: numpy array
            shape is (n_samples, 1), predicted probabilities
        file_path: string
            file path to save label probabilities in CSV
        """
        proba_df = pd.DataFrame({})
        for idx, label in self.idx_label_dic.items():
            proba_df[label] = pred_proba[:, idx]
        proba_df.to_csv(file_path, index=False)

    def init_all_models(self):
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
        
    def init_ensemble(self):
        all_models = [ensemble.AdaBoostClassifier(), ensemble.BaggingClassifier(), ensemble.ExtraTreesClassifier(),
                        ensemble.GradientBoostingClassifier(), ensemble.RandomForestClassifier()]
        self.models.extend(all_models)
        
    def init_linear(self):
        all_models = [linear_model.LogisticRegression()]
        self.models.extend(all_models)

    def init_naive_bayes(self):
        all_models = [naive_bayes.BernoulliNB(), naive_bayes.GaussianNB(), naive_bayes.MultinomialNB(), naive_bayes.ComplementNB()]
        self.models.extend(all_models)

    def init_neighbors(self):
        all_models = [neighbors.KNeighborsClassifier()]
        self.models.extend(all_models)

    def init_svm(self):
        all_models = [svm.NuSVC(probability=True), svm.SVC(probability=True)]
        self.models.extend(all_models)

    def init_decision_tree(self):
        all_models = [tree.DecisionTreeClassifier(), tree.ExtraTreeClassifier()]
        self.models.extend(all_models)