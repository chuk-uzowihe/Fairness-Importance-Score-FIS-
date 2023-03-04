import numpy as np
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from FIS import fis_tree, fis_forest, fis_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from FIS import util
import matplotlib.pyplot as plt

def I(x):
  indicators = x.astype(int)
  return indicators
    
        
def simulate_data_reg(nrow, ncol, alphas, betas, p, tau):
  """
  Simulates biased data.
  Inputs:
    * nrow - the number of rows in the output dataset.
    * ncol - the number of non-protected covariates X_i in the output dataset.
    * alphas - numpy array of scalars, where alpha_i controls the effect of
      protected attribute z on covariate X_j through the relationship
      X_i ~ Normal(alpha_i*z, 1).
    * betas - numpy array of scalars, where beta_j controls the effect of
      covariate X_i on the binary outcome y; can be thought of as the regression
      coefficients in the logistic regression scenario.
    * p - the probability of success (aka value 1) for our protected attribute
  Returns:
    * numpy array representing simulated tabular dataset 
  """

  # the following two assertions are to check that the input parameters
  # make sense - we should have one value of alpha and beta for each covariate
  assert ncol == len(alphas)
  assert ncol == len(betas)
  
  betas = np.reshape(betas, (len(betas),1))
  # z_i ~ Bernoulli(p)
  z = np.random.binomial(1, p, size=nrow)

  # initialize covariate matrix
  X = np.zeros((nrow, ncol))
  for j in range(ncol):
    X[:,j] =  np.random.normal(loc = alphas[j]*z,scale = 3, size = nrow)
  xb = (np.ones(nrow) + betas[0]*I(X[:,0] <tau) + betas[1]*I(X[:,1] <tau) + betas[3]*I(X[:,3] <tau) + 
        betas[4]*I(X[:,4] <tau) + betas[5]*I(X[:,5] <tau) + betas[6]*I(X[:,6] <tau) + 
        betas[7]*I(X[:,7] <tau) + betas[8]*I(X[:,8] <tau)+ 
       betas[9]*I(X[:,9] <tau) + betas[10]*I(X[:,10] <tau) + betas[11]*I(X[:,11] <tau) + 
       np.random.normal(loc = 0 , scale = 0.01, size = (nrow,1)))
  
  y_prob = expit(xb)
  
  
  # combine each element of dataset and we are all done!
  return z, X,y_prob


iterations = 10
nrow = 500
ncol = 12
value = 3
alphas = [3,3,3,0,0,0,3,3,3,0,0,0,]
beta_imp = 3*np.random.uniform(-1,1, size = 6)
beta_unimp = np.zeros(6)
betas = np.concatenate((beta_imp,beta_unimp))
p = 0.7


dp_mat = np.empty([iterations,ncol])
eo_mat = np.empty([iterations,ncol])
acc_mat = np.empty([iterations,ncol])

for i in range(iterations):
  
  z, X,y = simulate_data_reg(nrow,ncol ,alphas, betas, p, 0)
  clf = DecisionTreeRegressor()
  clf.fit(X,y)
  
  #Our approach
  f_forest = fis_tree(clf,X,y,z,0, regression = True)
  f_forest._calculate_fairness_importance_score()
  
  dp_mat[i] = f_forest._fairness_importance_score_dp
  eo_mat[i] = f_forest._fairness_importance_score_eqop
  acc_mat[i] = clf.feature_importances_



dp_mean = np.mean(dp_mat, axis = 0)
eo_mean = np.mean(eo_mat, axis = 0)
acc_mean = np.mean(acc_mat, axis = 0)

np.savetxt("/home/camille/FairFIS/reg_dp_tree_500.csv", dp_mean, delimiter= ",")


iterations = 10
nrow = 1000
ncol = 12
value = 3
alphas = [3,3,3,0,0,0,3,3,3,0,0,0,]
beta_imp = 3*np.random.uniform(-1,1, size = 6)
beta_unimp = np.zeros(6)
betas = np.concatenate((beta_imp,beta_unimp))
p = 0.7


dp_mat = np.empty([iterations,ncol])
eo_mat = np.empty([iterations,ncol])
acc_mat = np.empty([iterations,ncol])

for i in range(iterations):
  
  z, X,y = simulate_data_reg(nrow,ncol ,alphas, betas, p, 0)
  clf = DecisionTreeRegressor()
  clf.fit(X,y)
  
  #Our approach
  f_forest = fis_tree(clf,X,y,z,0, regression = True)
  f_forest._calculate_fairness_importance_score()
  
  dp_mat[i] = f_forest._fairness_importance_score_dp
  eo_mat[i] = f_forest._fairness_importance_score_eqop
  acc_mat[i] = clf.feature_importances_



dp_mean = np.mean(dp_mat, axis = 0)
eo_mean = np.mean(eo_mat, axis = 0)
acc_mean = np.mean(acc_mat, axis = 0)

np.savetxt("/home/camille/FairFIS/reg_dp_tree_1000.csv", dp_mean, delimiter= ",")






