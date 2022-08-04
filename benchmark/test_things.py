#%%
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

#%%
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
# %%
