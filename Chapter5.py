# ------------------------------------------------------------------------------
#                   Chapter 5: Support Vector Machines
# ------------------------------------------------------------------------------

# Soft-margin classification
from __future__ import division, print_function, unicode_literals
import os
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures


iris = datasets.load_iris()
X = iris["data"][:,(2,3)]       # petal length, petal width
y = (iris["target"]==2).astype(np.float64)

svm_classifier = Pipeline((
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ))

svm_classifier.fit(X, y)


# Non-linear classifiers
# Polynomial classifier (simple, no kernel)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

polynomial_svm_clf = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_classifier", LinearSVC(C=10, loss="hinge", random_state=42))
    ))

polynomial_svm_clf.fit(X, y)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

# Polynomial Kernel - NB the hyperparameter 'coef0' controls how much the model is influenced by high-degree polynomials
polynomial_kernel_svm_classifier = Pipeline((           # coef0=1 polynomial kernel
        ("scaler", StandardScaler()),
        ("svm_classifier", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ))
polynomial_kernel_svm_classifier.fit(X, y)

poly100_kernel_svm_classifier = Pipeline((              # coef0=100 polynomial kernel
        ("scaler", StandardScaler()),
        ("svm_classifier", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ))
poly100_kernel_svm_classifier.fit(X, y)

# plot the polynomial kernel classifiers with differnt r values
plt.figure(figsize=(11, 4))
plt.subplot(121)
plot_predictions(polynomial_kernel_svm_classifier, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)
plt.subplot(122)
plot_predictions(poly100_kernel_svm_classifier, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

plt.show()

# Gaussian Radial Bias Function (RBF) Kernel
rbf_kernel_classifier = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_classifier", SVC(kernel="rbf", gamma=5, C=0.001))
    ))
rbf_kernel_classifier.fit(X, y)

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_classifier = Pipeline((
            ("scaler", StandardScaler()),
            ("svm_classifier", SVC(kernel="rbf", gamma=gamma, C=C))
        ))
    rbf_kernel_classifier.fit(X, y)
    svm_clfs.append(rbf_kernel_classifier)

plt.figure(figsize=(11, 7))

for i, svm_classifier in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_classifier, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

plt.show()
# ------------------------------------------------------------------------------
# SVM Regression
np.random.seed(42)
m = 50
# new X and y data
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y)
