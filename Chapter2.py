import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pandas.plotting import scatter_matrix
# !!! make sure to explore all of these modules and their functions !!!

# Fetch and downlad data:
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url, housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Create a csv file containing the housing information fetched from the housing.tgz file
def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data(HOUSING_URL, HOUSING_PATH)
housing  = load_housing_data(HOUSING_PATH)
# print(housing.head(10))    # View the first 10 rows of the housing data
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
#
# housing.hist(bins=50, figsize=(9,9))
# plt.show()
np.random.seed(42)

def split_train_test(data, test_ratio):
    """This function splits the data set from the .csv file into a trining set and a
    test set (which should never
    be looked at!) according to a training-to-test set ratio"""
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print('Training set size: ', len(train_set))
print('Test set size: ', len(test_set))

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1]<256*test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()     # add an index column to the data
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
housing_with_id["id"] = housing["longitude"]*1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing["income_cat"].value_counts() / len(housing)

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Visualising data to gain some insights: Create a copy of the training data
housing = strat_train_set.copy()
# Create of scatter plot of the longitude and latitude information
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.show()

# Find correlations by computing the standard correlation coefficient (Pearson's r)
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# plot graphs of the correlation between select attributes using the scatter_matrix function from pandas
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "households"]
scatter_matrix(housing[attributes], figsize = (12,8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# Create attribute combinations to reveal additional insights
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Data preparation...
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
# To compensate for any missing features for each element, we have three options:
# 1) Get rid of the corresponding districts
# 2) Discard the whole attribute
# 3) Set the missing value to a 'filler' value (i.e. zero or the median for that feature
housing.dropna(subset=["total_bedrooms"])   # Option 1
housing.drop("total_bedrooms", axis=1)      # Option 2
median = housing["total_bedrooms"].median() # Option 3
housing["total_bedrooms"].fillna(median, inplace=True)  # Saves the median value when option 3 is used

# Scikitlean class Imputer can be used to fill missing values
imputer = Imputer(strategy="median")
# NB Imputer can only be used on numberical values, so a copy of the data without the text attribute ocean_proximity must be created
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)    # now fit the imputer instance to the training data
X = imputer.transform(housing_num)  # Transform training set by replacing missing values with the median values calculated NB training set is now a NumPy array!
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values))    # to convert from NumPy into a Pandas array
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# housing_tr.head()
# Encode ocean_proximity labels to numerical values using LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)

# Encode ocean_proximity labels to numerical values using OneHot
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))     # Stored as a SciPy sparse matrix
print(housing_cat_1hot)
housing_cat_1hot.toarray()      # converts to a NumPy array

# Both the transformations (text->int->onehot vector) LabelEncoder and OneHot can be applied in one shot with the LabelBinarizer:
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)       # Returns a dense NumPy array

# This class is used as a transformer that add the combined attributes created above: rooms_per_household, population_per_household, bedrooms_per_room
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """This class is used as a transformer that add the combined attributes: rooms_per_household, population_per_household, bedrooms_per_room"""
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", "population_per_household"])


# pipeline for transformations of numerical attruburtes of data
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

# Create a transformer class to handle Pandas DataFrames (since there is nothin in Scikitlean to handle this)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    """a transformer class to handle Pandas DataFrames (since there is nothin in Scikitlean to handle this)
    DataFrameSelector will trans data by selecting the desired attributes, dropping the rest, and then converting the resulting DataFrame into a NumPy array."""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Another pipeline for the categorical, rather than only the numerical, attributes. This pipeline would select only the categorical attributes using DataFrameSelector and LabelBinarizer
num_attribs = list(housing_num)   # Collect the numerical attributes
cat_attribs = ["ocean_proximity"]   # Collect the categorical attributes
# First create the pipeline for only numerical data:
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
# Now, a second pipeline for the categorical data:
# cat_pipeline = Pipeline([
#         ('selector', DataFrameSelector(cat_attribs))
#         ('label_binarizer', LabelBinarizer()),
#     ])      #Wrong
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])      #Correct

# Use FeatureUnion to join both of these pipelines into a single pipeline
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
# Now we have the finsihed, fully pre-processed data..
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)
# print(housing_prepared.shape)

# -------------------------------------------------------------------
#               Select and train the learning model
# -------------------------------------------------------------------
# Use the LinearRegression model from sklearn, trained on the prepared
# housing data from the pipeline and the respective housing labels:
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_prepared_data = full_pipeline.transform(some_data)
# print("Predictions: ", lin_reg.predict(some_prepared_data))
# print("Labels: ", list(some_labels))
predic = lin_reg.predict(some_prepared_data)
labs = list(some_labels)
print("Predictions: ", "    ", "Labels: ")
for elem in predic:
    print(elem)

# Measure the models RMSE to improve predicion accuracy
lin_reg_housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, lin_reg_housing_predictions)      # compute the mean squared error
lin_rmse = np.sqrt(lin_mse)     # compute the square root of the mse to get the RMSE
print("Linear Regression root mean squared error: ", lin_rmse)

# DecisionTreeRegressor regressor model:
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
tree_housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, tree_housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Decision tree regressor root mean squared error: ", tree_rmse)

# Time to evaluate the model using Cross-validation...
# The following section of code uses the Scikit-Learn cross validation features
# It performs  K-fold cross validation: splits the training set into 10 distinct subsets (folds)
# ...then trains and evaluates the decision tree model 10 times, picking a different fold for each evaluation
# result is an array containing the 10 evaliation scores:

# NB the Scikit-Learn cross validation methods expect a utility finction rather than a cost function!!

def display_scores(scores):
    """This function is used to display the scores from the Scikit-Learn K-fold cross validation process."""
    print("Scores: ")
    for i in scores:
        print(i)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("Linear Regression model: ")
display_scores(lin_rmse_scores)

tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
print("Decision Tree model: ")
display_scores(tree_rmse_scores)

# RandomForestRegressor model:
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

forest_housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, forest_housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Random Forest model root mean squared error: ", forest_rmse)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print("Random forest model: ")
display_scores(forest_rmse_scores)

# -------------------------------------------------------------------
#                   Fine-tune the learning model
# -------------------------------------------------------------------
# fiddling with the hyperparameters of the model manually is very tedious work...
# We use the Scikit-Learn GridSearchCV to search hyperparameters
# GridSearchCV is told which hyperparameters we want to experiment with and what values to try
# It then evaluates possible hyperparameter combinations using Cross-validation

# this section of code runs the GridSearchCV for the RandomForestRegressor model
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

print("Best hyperparameters: ", grid_search.best_params_)

# Obtain the best estimator directly:
print(grid_search.best_estimator_)
# Result:
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features=6, max_leaf_nodes=None, min_impurity_split=1e-07,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)


# Obtain the evaluation scores for the CV methods:
cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)

# Open the hood: observe the relative importance of each featue/attribute to help improve the accuracy of predictions
# Extract the weights of each of the features
feature_importance = grid_search.best_estimator_.feature_importances_
# Create a list of all of the features/attributes
extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted_features = sorted(zip(attributes, feature_importance), reverse=True)
# Print the features/attributes alongside their weights
# print("Attribute", "    ", "Feature importance")
# for attrib in sorted_features:
#     print(attrib)

# -------------------------------------------------------------------
#                   Evaluate the system on the test-set
# -------------------------------------------------------------------

# Establish the final model!
final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
percent_improvement = ((forest_rmse - final_rmse)/forest_rmse)*100
print("-------------------------------------------------------------------")
print("Root mean squared error of final model: ", final_rmse)
print("Percentage improement in RMSE: ", percent_improvement, "%")
print("-------------------------------------------------------------------")
