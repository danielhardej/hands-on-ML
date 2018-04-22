from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


mnist = fetch_mldata("MNIST original")

X, y = mnist["data"], mnist["target"]

some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
#
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
#
# label = y[36000]
# print(label)

# Separate the training data and the test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Shuffle the training data  (since some training algorithms are sensetive to the ordering of training data)
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Test out a binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# for i in y_train:
#     if y_train[i] == 5:
#         y_train_5 = y_train[i]
#     else:
#         pass
# return y_train_5
#
# for i in y_test:
#     if y_test[i] == 5:
#         y_test_5 = y_test[i]
#     else:
#         pass

sgd_classifier = SGDClassifier(random_state=42)
sgd_classifier.fit(X_train, y_train_5)

test_prediction = sgd_classifier.predict([some_digit])
print("SGD binary classifier test prediction: ", test_prediction)
if test_prediction == True:
    print("Test image correctly classified as 5")
else:
    print("Test image incorrectly classifed!")

# Evaluate the performace of the binary classifier using Cross-validation
skfolds = StratifiedKFold(n_splits = 3, random_state = 42)
# StratifiedKFold performs stratefied sampling of to produce folds that contain proportions of each class that are representative of the fill training set
for train_i, test_i in skfolds.split(X_train, y_train_5):
    clone_classifier = clone(sgd_classifier)
    X_train_folds = X_train[train_i]
    y_train_folds = (y_train_5[train_i])
    X_test_fold = X_train[test_i]
    y_test_fold = (y_train_5[test_i])

    clone_classifier.fit(X_train_folds, y_train_folds)
    y_pred = clone_classifier.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct, "out of", len(y_pred))

CV_score = cross_val_score(sgd_classifier, X_train, y_train_5, cv=3, scoring="accuracy")
print("Cross-validation score: ", CV_score)

# A dumb classifer to classify not-5s
class Not5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

not_5_classifier = Not5Classifier()
not_5_CV_score = cross_val_score(not_5_classifier, X_train, y_train_5, cv=3, scoring="accuracy")
print("Not-5 Cross-validation score: ", not_5_CV_score)

# NB a better way to evaluate the performace of a classifer is to use a CONFUSION MATRIX
y_train_pred = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3)

conf_mat = confusion_matrix(y_train_5, y_train_pred)
print(conf_mat)
# The confusion matrix gives rise to Precision and Recall scores
# precision = true_pos/(true_pos+false_pos)
# recall = true_pos/(true_pos+false_neg)
precision = precision_score(y_train_5, y_train_pred)
recall = recall_score(y_train_5, y_train_pred)
print("Precision: ", precision)
print("Recall: ", recall)
F_1 = (precision * recall)/(precision + recall)
print("F1 score:", F_1, f1_score(y_train_5, y_train_pred))

# observe the classifers decision function...
y_scores = sgd_classifier.decision_function([some_digit])
print(y_scores)
# But which threshold do we use? First we get the scores of all instances in the training set using cross_val_predict
y_scores = cross_val_predict(sgd_classifier, X_train, y_train_5, cv=3, method="decision_function")
# Then with these scores we compute the precision and recall for all possible thresholds using the precision_recall_curve function
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
# Finally we plot precision and recall as a function of threshold using matplotlib
def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])

# plot_precision_recall_threshold(precisions, recalls, thresholds)
# plt.show()
# We can use this plot to choose the threshold most suitable for our needs, based on what precision and recall values we'd need

# We can also plot precision against recall...
def plot_precision_recall(precisions, recalls):
    plt.plot(recalls, precisions, "g-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

# Say we want to aim for 90% precision. This would be from a threshold of > 70000
y_train_pred_90 = (y_scores > 70000)
pres_90 = precision_score(y_train_5, y_train_pred_90)
recall_90 = recall_score(y_train_5, y_train_pred_90)
print("Precision:", pres_90)
print("Recall: ", recall_90)

# The receiver operating characteristic (ROC) curve is onother tool used with binary classifiers. But instead of precision v recall,
# it plots the true positives v false positives
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    """This function produces the ROC curve - a plot of the true positive rate again the false positive rate -
    along side the ROC of a purely random classifier (the dotted line). Note that a perfect classifier will give a value of
    exactly 1 when the ROC curve is integrated."""
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],"k--")
    plt.axis([0,1,0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")

# plot_roc_curve(fpr, tpr)
# plt.show()

# NB a perfect classifier will give a value of 1 when the ROC curve is integrated; a purely random classifier will give a value of 0.5
# Lets evalute using the method - the area under the curve (AUC)
auc_val = roc_auc_score(y_train_5, y_scores)
print("Area under curve for SGD classifier: ", auc_val)

# Alright, lets try training a Random Forest classifier and compare the ROC curve and the ROC AUC to those of the SGD classifier
rf_classifier = RandomForestClassifier(random_state=42)
y_probas_rf = cross_val_predict(rf_classifier, X_train, y_train_5, cv=3, method="predict_proba")

# But wait, to plot an ROC curve we need scores, not probablities. We can use the positive class probability as the score!
y_scores_rf = y_probas_rf[:,1]      # score = prob of positive class
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_train_5, y_scores_rf)
# now we plot!
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_rf, tpr_rf, "Random Forest")
# plt.legend(loc="lower right")
# plt.show()

rf_auc_score = roc_auc_score(y_train_5, y_scores_rf)
# rf_precision = precision_score(y_train_5, y_probas_rf)
# rf_recall = recall_score(y_train_5, y_probas_rf)
print("Area under curve for Random Forest classifier: ", rf_auc_score)
# print("Random Forest precision: ", rf_precision)
# print("Random Forest recall: ", rf_recall)

# Now lets try a multi-class classifier!!
# Scale the features for later...
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# First train on the full X and y training sets:
sgd_classifier.fit(X_train, y_train)
multiclassSGD_test = sgd_classifier.predict([some_digit])
print("SGD multiclass test prediction: ", multiclassSGD_test)
some_digit_scores = sgd_classifier.decision_function([some_digit])
print("Decision function: ", some_digit_scores)
classes = sgd_classifier.classes_
print("Classes: ", classes)
SGD_multi_CV_score = cross_val_score(sgd_classifier,X_train, y_train, cv=3, scoring="accuracy")
SGD_multi_scaled_CV_score = cross_val_score(sgd_classifier, X_train_scaled, y_train, cv=3, scoring="accuracy")
print("Multiclass SGD CV score: ", SGD_multi_CV_score)
print("Multiclass SGD with scaling CV score: ", SGD_multi_scaled_CV_score)
# Lets force Scikit-Learn to use One vs One classification
ovo_classifier = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_classifier.fit(X_train, y_train)
ovo_test_prediction = ovo_classifier.predict([some_digit])
print("OvO test prediction: ", ovo_test_prediction)
n_ovo_estimators = len(ovo_classifier.estimators_)
print("Number of OvO estimators: ", n_ovo_estimators)

# Lets also try out a Random Forest classifier for multi-class predictions
rf_multi_classifier = RandomForestClassifier(random_state=42)
rf_multi_classifier.fit(X_train, y_train)
rf_multi_classifier_test = rf_multi_classifier.predict([some_digit])
print("Random Forest multi-class prediction: ", rf_multi_classifier_test)
rf_probs = rf_multi_classifier.predict_proba([some_digit])
print("Random Forest classifier probablities: ", rf_probs)

# Lets say we have decided on a promising model and are now looking for ways to tweak and imporve it. OK, time for Error Analysis!
# First we make some predictions:
y_train_pred = cross_val_predict(sgd_classifier, X_train_scaled, y_train, cv=3)
# generate the confusion matrix
conf_mat = confusion_matrix(y_train, y_train_pred)
print("The confusion matrix: ")
print(conf_mat)
# # we can make a plot of the confusion matrix too
# plt.matshow(conf_mat)
# plt.show()
# # Divide each value in the confusion matrix by the total number of images in the corresponding class to compare error rates
# row_sums = conf_mat.sum(axis=1, keepdims=True)
# norm_conf_mat = conf_mat/row_sums
# # Then we fill the diagonal of the matrix with zeros to show only the errors
# np.fill_diagonal(norm_conf_mat, 0)
# plt.matshow(norm_conf_mat, cmap=plt.cm.gray)
# plt.show()

# Now lets look at multilabel classification. To illustrate this, well make a classifier that identifies numbers that are both odd and more then or equal to 7
# This part of the code creates an array containing two target labels for each digit image: the first whether its large, the second whether its odd
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
# This part creates a K-Neighbors classifier, which supports multilabel classification (NB not all classifiers do!)
KNN_classifier = KNeighborsClassifier()
KNN_classifier.fit(X_train, y_multilabel)

knn_test = KNN_classifier.predict([some_digit])
print(knn_test)
# To evaluate this multilabel classifier, we compute an F1 score for each label and then average the scores (NB we can use other metrics for binary classifiers this way)
# y_train_knn_predict = cross_val_predict(KNN_classifier, X_train, y_train, cv=3)
# knn_f1 = f1_score(y_train, y_train_knn_predict, average="macro")
# print("Average F1 score for the K-Neighbors classifier: ", knn_f1)

# to wrap this up, lets look at Multi Output Classification. A generalization of multilabel classification, where each label can have two or more possible values.
# We'll test this out by building a system that removes noise from images.
# Start by making noisy images....
train_noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_noisy = X_train + train_noise
test_noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_noisy = X_test + test_noise
y_train_mod = X_train
y_test_mod = X_test
# Lets look at the noisy test images
some_noisy_digit = X_test_noisy[3600]
some_noisy_digit_image = some_noisy_digit.reshape(28, 28)
plt.imshow(some_noisy_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
# And now we'll train this multi output classifier with K-Neighbors
KNN_classifier.fit(X_train_noisy, y_train_mod)
some_clean_digit = KNN_classifier.predict([some_noisy_digit])
some_clean_digit_image = some_clean_digit.reshape(28, 28)
plt.imshow(some_clean_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
