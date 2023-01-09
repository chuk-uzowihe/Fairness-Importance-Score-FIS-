import numpy as np
from FIS.fis_null import fis_tree_null
from FIS.util import *
from FIS.base import fis_score
from FIS.fis import fis_tree
from joblib import Parallel, delayed
import sklearn.ensemble._forest as forest_utils
"""
A class to modify sklearn forests to calculate
FIS.
    ----------
    clf: Classifier
       The selector is a standard sklearn forest implementation
    train_x: nDarray of shape nxm
        The dataset for training
    y: nDarray of shape nx1:
        The true training labels
    protected_attribute: ndarray of shape nX1
        The protected feature
    protected_value: int, Default = 0
        Protected value of the protected attribute

    -----------
    Examples
    --------
    >>> from FIS import fis_forest
    >>> clf = tree.DecisionTreeClassifier()
    >>> f_forest = fis_forest(clf,train_x,train_y,z,0)
    >>> f_forest.fit(train_x, train_y)
    >>> f_forest.calculate_fairness_importance_score()
    >>> fis_dp = f_forest._fairness_importance_score_dp
    >>> fis_eqop = f_forest._fairness_importance_score_eqop
"""
class fis_forest(fis_score):
    def __init__(self,clf,train_x,train_y, protected_attribute, protected_value, normalize = True, regression = False, multiclass = False):
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
        self.normalize = normalize
        self.regression = regression
        self.multiclass = multiclass

    def fit(self, X, y):
        self.clf.fit(X,y)


    def predict(self, X):
        self.clf.predict(X)

    def predict_proba(self, X):
        self.clf.predict_proba(X)


    def _importance_for_each_tree(self,tree):
        '''
        n_samples = self.train_x.shape[0] # number of training samples

        n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(
            n_samples, rf.max_samples
        )

        unsampled_indices_trees = []
        sampled_indices_trees = []

        for estimator in rf.estimators_:
            unsampled_indices = forest_utils._generate_unsampled_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)
            unsampled_indices_trees.append(unsampled_indices)

            sampled_indices = forest_utils._generate_sample_indices(
                estimator.random_state, n_samples, n_samples_bootstrap)
            sampled_indices_trees.append(sampled_indices)
        '''
        
        individual_tree = fis_tree(tree, self.train_x, self.train_y, self.protected_attribute, self.protected_value, normalize=False, regression=self.regression, multiclass=self.multiclass)
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
        if self.normalize == True:
            self._fairness_importance_score_dp /= np.sum(abs(self._fairness_importance_score_dp))
            self._fairness_importance_score_eqop /= np.sum(abs(self._fairness_importance_score_eqop))
            self._fairness_importance_score_dp_root /= np.sum(abs(self._fairness_importance_score_dp_root))
            self._fairness_importance_score_eqop_root /= np.sum(abs(self._fairness_importance_score_eqop_root))
        

    def get_root_node_fairness(self):
        self.root_node_dp = np.zeros(self.number_of_features)
        self.root_node_eqop = np.zeros(self.number_of_features)
        self.feature_count = np.zeros(self.number_of_features)
        for estimator in self.fairness_estimators:
            dp, eqop, feature = estimator.get_root_node_fairness()
            self.root_node_dp[feature] += dp
            self.root_node_eqop[feature] += eqop
            self.feature_count[feature] += 1

        for i in range(self.number_of_features):
            self.root_node_dp[i] /= self.feature_count[i]
            self.root_node_eqop[i] /= self.feature_count[i]

        

