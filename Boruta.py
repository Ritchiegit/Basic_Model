"""
https://github.com/scikit-learn-contrib/boruta_py

"""
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = pd.read_csv('examples/test_X.csv', index_col=0).values
y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
print(X.shape)
print(y.shape)
y = y.ravel()
print(y.shape)
X = X[:, [9, 1, 2, 3, 4, 5, 6, 7, 8, 0]]
# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1, max_iter=30)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
print("feat_selector.support_", feat_selector.support_)
feat_selector_support_index = np.nonzero(feat_selector.support_)
print("feat_selector_support_index", feat_selector_support_index)
# check ranking of features
print("feat_selector.ranking_", feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)