"""
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
https://zhuanlan.zhihu.com/p/64900887
"""
# Boruta 和 RFE 接口确实很像
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
X, y = make_friedman1(n_samples=500, n_features=1000, random_state=0)  # 只有五个用于计算y，其余都和y无关
print("X.shape", X.shape)
print("y.shape", y.shape)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=20, step=1)
selector = selector.fit(X, y)
selector_support = selector.support_
selector_ranking = selector.ranking_
print(selector_support)
print(selector_ranking)

"""
https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_digits.html#sphx-glr-auto-examples-feature-selection-plot-rfe-digits-py
"""
"""
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
print(X.shape)
print(y.shape)
print(X)
print(y)
# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

# Plot pixel ranking
plt.matshow(ranking, cmap=plt.cm.Blues)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
"""