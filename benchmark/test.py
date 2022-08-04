#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
from FIS import fis

#%%
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#%%
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

