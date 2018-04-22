# ------------------------------------------------------------------------------
#                   Chapter 4: Training Models
# ------------------------------------------------------------------------------
# First we'll look at training the linear regression model by 1)  using a 'closed- form'
# equation to directly compute the best-fitting parameters, and 2) using the iterative
# optimization approch, Gradient Descent
from __future__ import division, print_function, unicode_literals       # Super important - does stuff
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn import datasets

np.random.seed(42)

# Generate some linear-looking training data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.rand(100, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline((
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler()),
    ))

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

def y_quad(x):
    y = (0.5*x**2) + x + 2
    return y

# Now we can try computing theta-hat (the value of theta that minimize the cost function)
X_b  = np.c_[np.ones((100, 1)), X]                              # Adds x_0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)    # Compute theta_best using the Normal Equation

print("Theta value after calculating using normal equation: ", theta_best)

# Okay, now lets try making a prediction
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print("Y prediction: ", y_predict)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.xlim(0, 2)
# plt.ylim(0, 15)
# plt.show()

# We could also use Scikit-Learn to create the same linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_int = lin_reg.intercept_
coeff = lin_reg.coef_
print(y_int)
print(coeff)
test_prediction = lin_reg.predict(X_new)
print(test_prediction)

# NB the normal equation is great for datasets with smaller numbers of features (around <100,000). Above this, the normal equation gets very slow
# as the number of features begins to get very large.

# So now we look at training the linear regression model using gradient descent...
# The general idea is to tweak paramters gradually until an optimal solution has been found (i.e. cost function has been minimized) - once the gradient reaches 0, we foundthe minimum!

# A quick, simple implementation of BATCH gradient descent o begin...
learning_rate = 0.1
n_iterations = 1000
m = 100                             # number of instances in the dataset (i.e. the training, cross val set etc.)
theta = np.random.randn(2, 1)       # random initialization of parameter values

for iteration in range(n_iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta)-y)
    theta = theta - learning_rate * gradients
    # print("Iteration: ", iteration, "Theta: ", theta)
print("BGD theta value: ", theta)

# Now lets look at stochastic gradient descent
theta_path_sgd = []
m = len(X_b)
np.random.seed(42)
n_epochs = 50
t0, t1 = 5, 50      # set the learning schedule hyper parameters for simulated annealing

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)       # Random initialization of parameter values

for epoch in range(n_epochs):
    for i in range(m):              # we iterate in rounds of m iterations - each of these rounds is an "epoch"!
        random_index = np.random.randint(m)
        x_i = X_b[random_index:(random_index+1)]
        y_i = y[random_index:(random_index+1)]
        gradients = (2/m) * x_i.T.dot(x_i.dot(theta)-y_i)
        learning_rate = learning_schedule(epoch * m + i)
        theta = theta - (learning_rate * gradients)
        theta_path_sgd.append(theta)
        # print("Epoch:", epoch, "iteration:", i, "theta val:", theta)
print("SGD theta values: ", theta)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# plt.show()

# To implement SGD using Scikit-Learn, you can use th SGDRegressor class. This uses the squared error to optimize
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
sklearn_sgd_intercept = sgd_reg.intercept_
sklearn_sgd_coeff = sgd_reg.coef_
print("intercept from sklearn SGD: ", sklearn_sgd_intercept)
print("coefficient from sklearn SGD: ", sklearn_sgd_coeff)

#  Mini-batch gradient descent
theta_path_mgd = []
n_iterations = 50
minibatch_size = 20
np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization
t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = (2/m) * xi.T.dot(xi.dot(theta) - yi)
        learning_rate = learning_schedule(t)
        theta = theta - (learning_rate * gradients)
        theta_path_mgd.append(theta)
print("Mini-batch GD theta val: ", theta)

# Cool. So what if the data set does not produce a straight line? What if it's more complex?
# We use POLYNOMIAL REGRESSION!
# Make some data that follows the shape of a quadratic equation curve
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])
# plt.show()

# Transform the data to add the 2nd degree polynomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# Create a linear regression model on this transformed data
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print("poly-reg coefficients: ", lin_reg.coef_, lin_reg.intercept_)

# To evalute whether a polynomial regression model is too simple or too complex, we can plot the learning curves...
# These are plots of the model performace on the training and cross validatio sets as a function of the training set size
def plot_learning_curves(model, X, y):
    """This function trains the polynomial regression model multiple times on different sized training sets, then
    plots the learning curves of the RSME performance."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])
plt.show()

polynomial_regression = Pipeline((
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ))

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])
plt.show()

# Regularization: to reduce over fitting, there are three key regularization techniques:
# Ridge regression, Lasso regression, and Elastic net.
# Closed-form implementation with Ridge regression using Scikit-Learn
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg_test = ridge_reg.predict([[1.5]])
# Closed form with Ridge regression using SGD
sgd_ridge_reg = SGDRegressor(penalty="l2")
sgd_ridge_reg.fit(X, y.ravel())
sgd_ridge_reg_test = sgd_reg.predict([[1.5]])
# Closed-form Lasso regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg_test = lasso_reg.predict([[1.5]])
# Lasso regression using SGD
sgd_lasso_reg = SGDRegressor(penalty="l1")
sgd_lasso_reg.fit(X, y.ravel())
sgd_lasso_reg_test = sgd_reg.predict([[1.5]])
# Elastic net regularization
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net_test = elastic_net.predict([[1.5]])

print("test: ", y_quad(1.5))
print("sklearn Ridge reg: ", ridge_reg_test, "SGD with ridge reg: ", sgd_ridge_reg_test)
print("Lasso regression: ", lasso_reg_test)
print("Elastic net: ", elastic_net_test)

# One simple way of regularizing iterative learning algorithmsis using Early Stopping -
# just stop the training once the vlidation error reaches a minimum!
# Here's a basic implementation
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)


# ------------------------------------------------------------------------------
#      Now time to look at somthing a little different: Logistic Regression
# ------------------------------------------------------------------------------
iris = datasets.load_iris()
iris_data_keys = list(iris.keys())
print(iris_data_keys)
#print(iris.DESCR)       # description of iris dataset

X = iris["data"][:,3:]
y = (iris["target"]==2).astype(int)

log_reg = LogisticRegression()
log_reg.fit(X, y)

# Look at the model estimated probability for flowers with petal widths of 0-3cm
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:,1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:,0], "b--", label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()

print("Decision boundary:", decision_boundary)

# To make our classifier more confident in its predictions, we can use more features other than only petal width!
# We can use length too! We can train a Logistic Regression classifier on width and length
# First, get the new data:
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
plt.show()

# Finally, a way to use Logistic Regression in classifying multiple classes directly: Softmax Regression
#
X = iris["data"][:,(2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap, linewidth=5)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

# test it out:
softmax_test = [5, 2]       # 5cm long, 2cm wide petals; an Iris-Virginica

softmax_prediction_result = softmax_reg.predict([softmax_test])
softmax_prediction_probability = np.amax((softmax_reg.predict_proba([softmax_test]))*100)

if softmax_prediction_result == 0:
    predicion = "Iris-Setosa"
elif softmax_prediction_result == 1:
    prediction = "Iris-Versicolor"
elif softmax_prediction_result == 2:
    prediction = "Iris-Virginica"
else:
    pass

print("Softmax prediction test result:", prediction)
print("Softmax prediction test result probability:", softmax_prediction_probability, "%")
