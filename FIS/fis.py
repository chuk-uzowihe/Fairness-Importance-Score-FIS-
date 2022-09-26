import numpy as np

from FIS.base import fis_score
from FIS.util import *

class fis_tree():
    def __init__(self, fitted_clf,train_x,train_y, protected_attribute, protected_value):
        self.fitted_clf = fitted_clf
        self.train_x = train_x
        self.train_y = train_y
        self.protected_attribute = protected_attribute
        self.protected_value = protected_value
        self.samples_at_node = {}
        self.eqop_at_node = {}
        self.dp_at_node = {}
        self.number_of_features = train_x.shape[1]
        self._fairness_importance_score_dp = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop = np.zeros(self.number_of_features)
        self.children_left = fitted_clf.tree_.children_left
        self.children_right = fitted_clf.tree_.children_right
        self.feature = fitted_clf.tree_.feature
        self.n_nodes = fitted_clf.tree_.node_count
        self.number_of_samples = fitted_clf.tree_.weighted_n_node_samples
        self._fairness_importance_score_dp_root = np.zeros(self.number_of_features)
        self._fairness_importance_score_eqop_root = np.zeros(self.number_of_features)

    def _calculate_fairness_importance_score(self):
        path = self.fitted_clf.decision_path(self.train_x)
        samples = path.shape[0]
        [self.samples_at_node.setdefault(i, []) for i in range(self.n_nodes)]
        for i in range(samples):
            for(j) in range(self.n_nodes):
                if path[i, j] != 0:
                    #print(i,j)
                    self.samples_at_node[j].append(i)
        self.calculate_fairness_at_each_node()
        self.calculate_fairness_importance_score()
        

    def calculate_fairness_at_each_node(self):
        self.train_x_with_protected = np.concatenate((self.train_x,np.reshape(self.protected_attribute,(-1,1))),axis=1) 
        
        for n in range(self.n_nodes):
            if self.children_left[n] != self.children_right[n]:
                left = self.samples_at_node[self.children_left[n]]
                right = self.samples_at_node[self.children_right[n]]
                X_left = self.train_x_with_protected[left]
                X_right = self.train_x_with_protected[right]
                y_left = self.train_y[left]
                y_right = self.train_y[right]
                if n == 0:
                    self.eqop_at_node[0] = 1 - fairness_rndm(X_left, y_left, X_right, y_right,self.number_of_features,0,1)
                    self.dp_at_node[0] = 1 - fairness_rndm(X_left, y_left, X_right, y_right,self.number_of_features,0,2)
                    #self.eqop_at_1 = 1 - previous_fairness(X_left, y_left, X_right, y_right,self.number_of_features,0,1)
                    #self.dp_at_1 = 1 - previous_fairness(X_left, y_left, X_right, y_right,self.number_of_features,0,2)
                self.eqop_at_node[self.children_left[n]] = self.eqop_at_node[self.children_right[n]] = fairness(X_left, y_left, X_right, y_right,self.number_of_features,0,1)
                self.dp_at_node[self.children_left[n]] = self.dp_at_node[self.children_right[n]] = fairness(X_left, y_left, X_right, y_right,self.number_of_features,0,2)
                #self.eqop_at_node[self.children_left[n]] = 1 - fairness_rndm(X_left, y_left, X_right, y_right,self.number_of_features,0,1)
                #self.dp_at_node[self.children_left[n]] = 1 - fairness_rndm(X_left, y_left, X_right, y_right,self.number_of_features,0,1)
    
        ##self.eqop_at_node[0] = 1 
        #self.dp_at_node[0] = 1 

    def calculate_fairness_importance_score(self):
        for i in range(self.n_nodes):
            if i == 0 and self.children_right[i] != self.children_left[i]:
                self._fairness_importance_score_dp_root[self.feature[i]] += \
                    ((self.dp_at_node[self.children_left[i]] - self.dp_at_node[i])\
                        /len(self.samples_at_node[0]))
                self._fairness_importance_score_eqop_root[self.feature[i]] \
                    += ((self.eqop_at_node[self.children_left[i]] - self.eqop_at_node[i])\
                        *len(self.samples_at_node[i])/len(self.samples_at_node[0]))
            if self.children_right[i] != self.children_left[i] and i != 0:
                self._fairness_importance_score_dp[self.feature[i]] += ((self.dp_at_node[self.children_left[i]] - self.dp_at_node[i])*len(self.samples_at_node[i])/len(self.samples_at_node[0]))
                self._fairness_importance_score_eqop[self.feature[i]] \
                    += ((self.eqop_at_node[self.children_left[i]] - self.eqop_at_node[i])\
                        *len(self.samples_at_node[i])/len(self.samples_at_node[0]))
                self._fairness_importance_score_dp_root[self.feature[i]] += ((self.dp_at_node[self.children_left[i]] - self.dp_at_node[i])*len(self.samples_at_node[i])/len(self.samples_at_node[0]))
                self._fairness_importance_score_eqop_root[self.feature[i]] \
                    += ((self.eqop_at_node[self.children_left[i]] - self.eqop_at_node[i])\
                        *len(self.samples_at_node[i])/len(self.samples_at_node[0]))
    
    def get_root_node_fairness(self):
        return self.dp_at_node[1], self.eqop_at_node[1], self.feature[0]





    


