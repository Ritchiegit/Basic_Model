# docx https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# example
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_dense_vs_sparse_data.html#sphx-glr-auto-examples-linear-model-plot-lasso-dense-vs-sparse-data-py
# https://machinelearningmastery.com/lasso-regression-with-python/

# from sklearn import linear_model
# clf = linear_model.Lasso(alpha=0.1)
# clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
# print(clf.coef_)
# print(clf.intercept_)
# y_pred = clf.predict([[0, 0], [1, 1]])
# print(y_pred)
# print(clf.get_params())
"""
# load and summarize the housing dataset
from sklearn import linear_model
from pandas import read_csv
from matplotlib import pyplot
# load dataset
url = "data/housing.csv"  # 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
# summarize shape
print(dataframe.shape)
# summarize first few lines
# print(dataframe.head())
model = linear_model.Lasso(alpha=1.0)
"""

# evaluate an lasso regression model on the dataset
from numpy import mean
from numpy import std
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
# load the dataset
url = "data/housing.csv"  # 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Lasso(alpha=1.0)
"""
# cross validation
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
# MAE
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
"""
# fit and predict
model.fit(X, y)
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
y_pred = model.predict([row, row])
print("y_pred", y_pred)

# Tuning Lasso Hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from numpy import arange
from pandas import read_csv
from sklearn.linear_model import Lasso

url = "data/housing.csv"
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
model = Lasso()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid["alpha"] = arange(0, 1, 0.01)
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
results = search.fit(X, y)
print("MAE: %.3f" % results.best_score_)
print("Config: %s" % results.best_params_)

from numpy import arange
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
model.fit(X, y)
print(model.alpha_)
