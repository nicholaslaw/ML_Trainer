from MLTrainer import MLTrainer
import numpy as np
from config import *

X = np.random.normal(size=(1000, 3))
Y = np.random.randint(2, size=1000)

models = MLTrainer()
models.fit(X, Y, n_jobs=-1, gridsearchcv=True, param_grids=classf_grids)
print(models.models)
models.evaluate(X, Y, None)