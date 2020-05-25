from MLTrainer import MLTrainer
import numpy as np

X = np.random.normal(size=(50, 2))
Y = np.random.randint(2, size=50)

models = MLTrainer()
print(models.get_models_scores(X, Y, n_jobs=-1, gridsearchcv=False, param_grids=None))
models.evaluate(X, Y, None)