import numpy as np
from FIS.util import *
from FIS.base import fis_score
from FIS.fis import fis_tree
from joblib import Parallel, delayed

class fis_forest(fis_score):
    def __init__(self,clf,train_x,train_y, protected_attribute, protected_value):
        self.clf = clf
        self.train_x = train_x
        self.train_y = train_y
        self.protected_attribute = protected_attribute
        self.protected_value = protected_value
        self.number_of_features = train_x.shape[1]
        self._fairness_importance_score_dp = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop = np.zeros(self.number_of_features)
        self._fairness_importance_score_dp_root = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop_root = np.zeros(self.number_of_features)

    def fit(self, X, y):
        self.clf.fit(X,y)


    def predict(self, X):
        self.clf.predict(X)

    def predict_proba(self, X):
        self.clf.predict_proba(X)


    def _importance_for_each_tree(self,tree):
        individual_tree = fis_tree(tree, self.train_x, self.train_y, self.protected_attribute, self.protected_value)
        individual_tree._calculate_fairness_importance_score()
        
        return individual_tree

    def calculate_fairness_importance_score(self):
        self.fairness_estimators = []
        self.fairness_estimators = Parallel(n_jobs=-2,verbose=1)(
        delayed(self._importance_for_each_tree)(tree) 
        for tree in self.clf.estimators_
        )
        for individual_tree in self.fairness_estimators:
            for i in range(self.number_of_features):
                self._fairness_importance_score_dp[i] += individual_tree._fairness_importance_score_dp[i]
                self._fairness_importance_score_eqop[i] += individual_tree._fairness_importance_score_eqop[i]
                self._fairness_importance_score_dp_root[i] += individual_tree._fairness_importance_score_dp_root[i]
                self._fairness_importance_score_eqop_root[i] += individual_tree._fairness_importance_score_eqop_root[i]
        self._fairness_importance_score_dp /= len(self.clf.estimators_)
        self._fairness_importance_score_eqop /= len(self.clf.estimators_)
        self._fairness_importance_score_dp_root /= len(self.clf.estimators_)
        self._fairness_importance_score_eqop_root /= len(self.clf.estimators_)
