"""
# 20201227
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

"""
# load data
from sklearn import datasets
print(datasets)

cancer = datasets.load_breast_cancer()

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# explore data
print(cancer)
print(cancer.data)
print(cancer.data.shape)
print(cancer.data[0:5])
print(cancer.target[0:5])
print("*" * 20)

# data split
from sklearn.model_selection import train_test_split
d = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
X_train, X_test, y_train, y_test = d
# 处理成数组传入
print(d)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 直接给划分出了数据集
# 可以直接使用 位置进行划分
# TODO dataloader

from sklearn import svm

clf = svm.SVC(kernel='linear')

print(clf)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
"""
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# metrics
from sklearn import metrics
print("AUC:", metrics.roc_auc_score(y_test, y_pred))
print(f"Accuracy:{metrics.accuracy_score(y_test, y_pred)}")
print(f"average_precision_score:{metrics.average_precision_score(y_test, y_pred)}")
print(f"adjusted_mutual_info_score:{metrics.adjusted_mutual_info_score(y_test, y_pred)}")
print(f"adjusted_rand_score:{metrics.adjusted_rand_score(y_test, y_pred)}")
print(f"balanced_accuracy_score:{metrics.balanced_accuracy_score(y_test, y_pred)}")
print(f"f1_score:{metrics.f1_score(y_test, y_pred)}")
print(f"mean_absolute_error:{metrics.mean_absolute_error(y_test, y_pred)}")
print(f"mean_squared_error:{metrics.mean_squared_error(y_test, y_pred)}")


