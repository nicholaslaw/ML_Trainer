import numpy as np
from sklearn.model_selection import StratifiedKFold

class LearningCurve:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size

    def plot(self, model, train_X: Union[tuple, list, np.ndarray], train_Y: Union[tuple, list, np.ndarray], test_X: Union[tuple, list, np.ndarray], test_Y: Union[tuple, list, np.ndarray]):
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

        RETURNS
        ==========
        plots learning curve for model
        """
        pass