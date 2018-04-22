# ------------------------------------------------------------------------------
#                           Chapter 6: Decision Trees
# ------------------------------------------------------------------------------

from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz

np.random.seed(42)
PROJECT_ROOT_DIR = "/Users/danielhardej/Documents/Geron_HandsOn_ML"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, fig_id)

# Create a simple Decision Tree classifier
iris = load_iris()
X = iris.data[:, 2:]        # Petal length and width
y = iris.target

tree_classifier = DecisionTreeClassifier(max_depth=2)
tree_classifier.fit(X, y)

export_graphviz(
        tree_classifier,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

# Test out a prediction using the decision tree model on a leaf with 5cm length and 1.5cm width
test_petal = [5, 1.5]       # 5cm length, 1.5cm width
test_prediction = tree_classifier.predict([test_petal])
test_prediction_proba = tree_classifier.predict_proba([test_petal])

if test_prediction == 0:
    prediction = "Iris-Setosa"
elif test_prediction == 1:
    prediction = "Iris-Versicolor"
elif test_prediction == 2:
    prediction = "Iris-Virginica"
else:
    pass

print("Test prediction:", prediction)
print("Test prediction probability:", test_prediction_proba)

# Now lets try a Decision Tree for a regression task
# First, generate some noisy quadratic data
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10
# Then create the Decision Tree regresssion model:
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42, min_samples_leaf=10)
tree_reg.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    for split in (0.0458, 0.1298, 0.2873, 0.9040):
        plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)

plot_regression_predictions(tree_reg, X, y)
plt.show()
