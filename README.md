# FARC-HD for Scikit-Learn
=========================

**FARC-HD** (*Fuzzy Association Rule-based Classifier for High-Dimensional problems*) is a fuzzy logic-based classifier designed to balance **interpretability** and **accuracy** in complex datasets.

This implementation is fully compatible with the **Scikit-Learn** API, allowing for seamless integration into `Pipelines`, `GridSearchCV`, and standard cross-validation workflows.

### ✨ Key Features
* **Interpretability:** Generates human-readable linguistic rules to explain model decisions.
* **High-Dimensional Efficiency:** Leverages the **Apriori** algorithm for rule extraction, optimized for large feature sets.
* **Evolutionary Optimization:** Includes a genetic algorithm stage to tune rule weights and improve generalization.
* **High Performance:** Critical inference components are accelerated using **Numba (JIT compilation)** for near-native execution speeds.

### 🚀 Quick Start
```python
from farc_hd import FarcHDClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and fit
clf = FarcHDClassifier(n_labels=5, depth=3)
clf.fit(X_train, y_train)

# Predict and inspect rules
y_pred = clf.predict(X_test)
clf.print_rules()