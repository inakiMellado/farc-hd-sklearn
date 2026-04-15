"""
======================================
Plotting FARC-HD Classifier
======================================

An example showing the decision boundary of the :class:`farc_hd.FarcHDClassifier`
on a 2D slice of the Iris dataset.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from farc_hd.FarcHDClassifier import FarcHDClassifier

# 1. Cargamos datos (usamos solo 2 características para poder dibujarlo en 2D)
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# 2. Instanciamos el clasificador FARC-HD con parámetros ligeros
# para que la compilación de la documentación sea rápida.
clf = FarcHDClassifier(
    n_labels=3,
    depth=2,
    max_trials=20,
    population_size=5
)

# 3. Entrenamos el modelo
clf.fit(X, y)

# 4. Dibujamos la frontera de decisión
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict", alpha=0.5, cmap=plt.cm.coolwarm
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
disp.ax_.set_title("FARC-HD Decision Boundary (Iris 2D)")
plt.show()