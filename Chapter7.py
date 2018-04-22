# ------------------------------------------------------------------------------
#                 Chapter 7: Ensmble Learning and Random Forests
# ------------------------------------------------------------------------------
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(42)

# Lets try a simple Ensemble/Voting classifier, using three different classifiers, and trained on the 'moons' dataset
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# The three independant classifiers used are: Logistic Regression, Random Forest, and a Support Vector Machine Classifier
logistic_classifier = LogisticRegression()
random_forest_classifier = RandomForestClassifier()
svm_classifier = SVC(probability=True)
# The Voting (Ensemble) classifier will be an amalgamation of these three i.e. the all classify, then vote. The final classification is the result of that vote.
voting_classifier = VotingClassifier(
    estimators=[("lr", logistic_classifier), ("rf", random_forest_classifier), ("svc", svm_classifier)],
    voting="soft")
voting_classifier.fit(X_train, y_train)

# Test the classifier and find out the accuracy of each of the individual classifiers
for clf in (logistic_classifier, random_forest_classifier, svm_classifier, voting_classifier):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Ensemble Method 1: Bagging and Pasting
bag_classifier = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1)
bag_classifier.fit(X_train, y_train)
y_pred = bag_classifier.predict(X_test)

tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)
y_pred_tree = tree_classifier.predict(X_test)

print("Bag classifier accuracy: ", accuracy_score(y_test, y_pred))
print("Single Decision tree classifier accuracy: ", accuracy_score(y_test, y_pred_tree))

# We can use the 'oob' score to evaluate how well a Baggin Classifier performs
# Create a Bagging Classifier that records to oob-score
bag_classifier_with_oob = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True)
bag_classifier_with_oob.fit(X_train, y_train)

oob = bag_classifier_with_oob.oob_score_
y_pred_oob = bag_classifier_with_oob.predict(X_test)
oob_accuracy = accuracy_score(y_test, y_pred_oob)

print("Bag classifier oob-evaluation: ", oob)
print("Bag classifier (with oob) accuracy: ", oob_accuracy)

# Now a look at Random Forests - an ensemble of Decision Trees
# Equivalent to a Bagging Classifier full of Decision Tree classifiers with splitter="random"
random_forest_classifier = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
# random_forest_classifier.fit(X_train, y_train)
#
# y_pred_rf = random_forest_classifier.predict(X_test)

# The Random Forest classifier also lets us observe the relative importance of each of the features
iris = load_iris()
random_forest_classifier.fit(iris["data"], iris["target"])
print("Relative importance of each feature in iris dataset according to Random Forest classifier: ")
for name, score in zip(iris["feature_names"], random_forest_classifier.feature_importances_):
    print(name, score)

# Ensemble Method 2: Boosting
# There are two types commonly used: AdaBoost and Gradient Boost
# First, AdaBoost (with 200 Decision Stumpms. A Decision Stump is a Decision Tree with a max_depth=1, equivalent to a single decision node and two leafs)
ada_classifier = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5)
ada_classifier.fit(X_train, y_train)
# NB there's also an AdaBoostRegressor!

# Now Gradient Boosting
# Make some noisy quadratic data...
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X, y)
# Following code finds the optimal number of trees using early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y)
gbrt = GradientBoostingRegressor(nax_depth=2, n_estimators=120)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
        for y_pred in gbrt.staged_predict(X_val)]
best_n_estimators = np.argmin(errors)

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimators)
gbrt_best.fit(X_train, y_train)

# Plot the Validation error against the number of trees and also the best model (with the corresponding number of trees)
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")
plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

save_fig("early_stopping_gbrt_plot")
plt.show()

# It is also possible to implement early stopping by actually stopping early (rather than training a lot of trees and then picking the best one)
# This is done by setting warm_start=True
# This code stops the training when error does not improve for 5 consecutive iterations (similar to optimizing according to some residual value)
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break       # induces early stopping!

# Ensemble Method 3: Stacking
# There's nothing for us to do here. Maybe another time.
