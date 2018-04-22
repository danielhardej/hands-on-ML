# ------------------------------------------------------------------------------
#                      Chapter 8: Dimensionality Reduction
# ------------------------------------------------------------------------------
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA

from sklearn.manifold import LocallyLinearEmbedding

from sklearn.metrics import mean_squared_error

from sklearn.datasets import make_swiss_roll

np.random.seed(42)

# ------------------------------ Vanilla PCA -----------------------------------
# Incremental PCA:

# Randomized PCA:


# ------------------------------ Kernel PCA ------------------------------------
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)              # Create the data
y = t > 6.9

rbf_kernel_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)          # Create the Kernel PCA with an RBF Kernel
X_reduced = rbf_kernel_pca.fit_transform(X)                                     # Transform the data with the RBF Kernel PCA

lin_kernel_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)                          # Linear Kernel
rbf_kernel_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)               # RBF Kernel
sig_kernel_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)   # Sigmoid Kernel

# How to select a Kernel and tune hyperparameters: use grid search to select the
# kernel and hyperparameters that lead to the best performace for the given task
# Do this by creating a two-step pipeline: Step 1) Reduce dimensionality to two
# dimensions using kPCA, and then Step 2) apply Logistic Regression for
# classification.

classifier = Pipeline([
                ("kpca", KernelPCA(n_components=2)),
                ("log_reg", LogisticRegression())
            ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(classifier, param_grid, cv=3)
grid_search.fit(X, y)

print(grid_search.best_params_)

# Computing Reconstuction Error: An alternative to using Grid Search for finding
# the best kernel to use and tuning hyperparameters.
rbf_kernel_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
X_reduced = rbf_kernel_pca.fit_transform(X)
X_preimage = rbf_kernel_pca.inverse_transform(X_reduced)

reconstruction_error = mean_squared_error(X, X_preimage)

# This can be used with Grid Search and Cross-Validation to find the kernel and
# hyperparameters that minimize the pre-image reconstruction error.

# -------------------- Locally Linear Embedding (LLE) --------------------------
# A Manifold Learning technique that does not rely on projections - particularly
# good at unrolling twisted manifolds.
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)

plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
plt.show()
