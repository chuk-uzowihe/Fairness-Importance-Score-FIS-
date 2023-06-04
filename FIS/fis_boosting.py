import numpy as np

from FIS.base import fis_score
from FIS.util import *
from FIS.fis import fis_tree
from joblib import Parallel, delayed



class fis_boosting():
    def __init__(self, fitted_clf,train_x,train_y, protected_attribute, protected_value, normalize = True, regression = False, multiclass = False,triangle = True):
        self.fitted_clf = fitted_clf
        self.train_x = train_x
        self.train_y = train_y
        self.protected_attribute = protected_attribute
        self.protected_value = protected_value
        self.number_of_features = train_x.shape[1]
        self._fairness_importance_score_dp = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop = np.zeros(self.number_of_features)
        self._fairness_importance_score_dp_root = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop_root = np.zeros(self.number_of_features)
        self.individual_feature_values = {}
        self.normalize = normalize
        self.regression = regression
        self.multiclass = multiclass
        self.triangle = True


    def each_tree(self,index):
        individual_tree = fis_tree(self.fitted_clf.estimators_[index,0], self.train_x, self.train_y, self.protected_attribute, self.protected_value, normalize = False, regression=self.regression, multiclass = self.multiclass, triangle=self.triangle)
        individual_tree._calculate_fairness_importance_score()
        return individual_tree
    
    def calculate_fairness_importance_score(self):
        [self.individual_feature_values.setdefault(i, []) for i in range(self.number_of_features)]
        self.trees = []
        self.train_x_with_protected = np.concatenate((self.train_x,np.reshape(self.protected_attribute,(-1,1))),axis=1) 
        #self.dp_pred = 1 - DP(self.train_x_with_protected,self.train_y,self.fitted_clf.predict(self.train_x), self.number_of_features,0)
        #self.eq_pred = 1 - eqop(self.train_x_with_protected,self.train_y,self.fitted_clf.predict(self.train_x), self.number_of_features,0)
        
        self.trees = Parallel(n_jobs=-2,verbose=1)(
        delayed(self.each_tree)(index) 
        for index in range(self.fitted_clf.n_estimators_)
        )
        
        for individual_tree in self.trees:
            #individual_tree = fis_tree(self.fitted_clf.estimators_[i,0], self.train_x, self.train_y, self.protected_attribute, self.protected_value)
            #individual_tree._calculate_fairness_importance_score()
            #self.trees.append(individual_tree)
            dp, eq, feature = individual_tree.get_root_node_fairness()
            null_dp, null_eq, feature_null = individual_tree.get_null_fairness()
            self.individual_feature_values[feature].append((dp,eq,null_dp,null_eq))
            for i in range(self.number_of_features):
                self._fairness_importance_score_dp[i] += individual_tree._fairness_importance_score_dp[i]
                self._fairness_importance_score_eqop[i] += individual_tree._fairness_importance_score_eqop[i]
                self._fairness_importance_score_dp_root[i] += individual_tree._fairness_importance_score_dp_root[i]
                self._fairness_importance_score_eqop_root[i] += individual_tree._fairness_importance_score_eqop_root[i]
        self._fairness_importance_score_dp /= (self.fitted_clf.n_estimators_)
        self._fairness_importance_score_eqop /= (self.fitted_clf.n_estimators_)
        if self.normalize == True:
            self._fairness_importance_score_dp /= np.sum(abs(self._fairness_importance_score_dp))
            self._fairness_importance_score_eqop /= np.sum(abs(self._fairness_importance_score_eqop))
            self._fairness_importance_score_dp_root /= np.sum(abs(self._fairness_importance_score_dp_root))
            self._fairness_importance_score_eqop_root /= np.sum(abs(self._fairness_importance_score_eqop_root))

    def calculate_fairness_importance_score_nonstamp(self):
        w = np.zeros(self.fitted_clf.n_estimators_)
        
        self.number_of_estimators = self.fitted_clf.n_estimators_
        self.trees = []
        self.train_x_with_protected = np.concatenate((self.train_x,np.reshape(self.protected_attribute,(-1,1))),axis=1) 
        self.dp_pred = 1 - DP(self.train_x_with_protected,self.train_y,self.fitted_clf.predict(self.train_x), self.number_of_features,0)
        self.eq_pred = 1 - eqop(self.train_x_with_protected,self.train_y,self.fitted_clf.predict(self.train_x), self.number_of_features,0)
        
        self.trees = Parallel(n_jobs=-2,verbose=1)(
        delayed(self.each_tree)(index) 
        for index in range(self.fitted_clf.n_estimators_)
        )
        tree = self.fitted_clf.estimators_[0, 0].tree_
        leaf_mask = tree.children_left == -1  # TREE_LEAF == -1
        w_i = tree.value[leaf_mask, 0, 0]
       
       
        #for i in range(self.fitted_clf.n_estimators_):
        #    individual_tree = fis_tree(self.fitted_clf.estimators_[i,0], self.train_x, self.train_y, self.protected_attribute, self.protected_value)
        #    individual_tree._calculate_fairness_importance_score()
        #    self.trees.append(individual_tree)
        for individual_tree in self.trees:    
            for i in range(self.number_of_features):
                self._fairness_importance_score_dp[i] += individual_tree._fairness_importance_score_dp[i]
                self._fairness_importance_score_eqop[i] += individual_tree._fairness_importance_score_eqop[i]
                self._fairness_importance_score_dp_root[i] += individual_tree._fairness_importance_score_dp_root[i]
                self._fairness_importance_score_eqop_root[i] += individual_tree._fairness_importance_score_eqop_root[i]
        self._fairness_importance_score_dp /= self.number_of_estimators
        self._fairness_importance_score_eqop /= self.number_of_estimators
        self._fairness_importance_score_dp_root /= self.number_of_estimators
        self._fairness_importance_score_eqop_root /= self.number_of_estimators