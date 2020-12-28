"""
# 20201228
https://stackabuse.com/implementing-lda-in-python-with-scikit-learn/
接口 x 2
https://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
一个例子
https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html

"""

import pandas as pd
import numpy as np
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data_path = "data/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(data_path, names=names)
# print(dataset)


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values  # 最后一维是标签
#
# print(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""
random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Performing LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)  # 在这里进行训练
X_test = lda.transform(X_test)

# training and prediction
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy{accuracy_score(y_test, y_pred)}')
