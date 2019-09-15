from MLTrainer import MLTrainer
import numpy as np

X = np.random.normal(size=(1000, 3))
Y = np.random.randint(2, size=1000)

models = MLTrainer()
models.fit(X, Y)
print(models.models)
models.evaluate(X, Y, None)