#%%
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
#%%
def I(x):
  indicators = x.astype(int)
  return indicators
    
#%%        
def simulate_data(nrow, ncol, alphas, betas, p, tau, seed, regression = False):
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
    np.random.seed(seed)
    assert ncol == len(alphas)
    assert ncol == len(betas)
    
    betas = np.reshape(betas, (len(betas),1))
    # z_i ~ Bernoulli(p)
    z = np.random.binomial(1, p, size=nrow)

    # initialize covariate matrix
    X = np.zeros((nrow, ncol))
    for j in range(ncol):
       X[:,j] =  alphas[j]*z + np.random.normal(loc = 0 , scale = 1, size = nrow)
    xb = (np.ones(nrow) + betas[0]*I(X[:,0] <tau) + betas[1]*I(X[:,1] <tau) + betas[3]*I(X[:,3] <tau) + 
          betas[4]*I(X[:,4] <tau) + betas[5]*I(X[:,5] <tau) + betas[6]*I(X[:,6] <tau) + 
          betas[7]*I(X[:,7] <tau) + betas[8]*I(X[:,8] <tau)+ 
        betas[9]*I(X[:,9] <tau) + betas[10]*I(X[:,10] <tau) + betas[11]*I(X[:,11] <tau))
    if regression == True:
       xb = X@betas
       y = xb + np.random.normal(0,0.1,nrow)
       return z,X,y
    y_prob = expit(xb)
    
    y = np.zeros(nrow)
    for i in range(len(y_prob)):
        y[i] = np.random.binomial(1,y_prob[i])
    
    # combine each element of dataset and we are all done!
    return z, X,y

#%%
iterations = 25
nrow = 500
ncol = 12
alphas = [1,1,1,0,0,0,1,1,1,0,0,0]
#beta_imp = 3*np.random.uniform(-1,1, size = 6)
beta_imp = np.zeros(6)
np.random.seed(1000)
for i in range(6):
  p = np.random.binomial(1,0.5,1)
  if p == 1:
      value = np.random.uniform(3,3)
  else:
      value = np.random.uniform(3,3)
  
  beta_imp[i] = value


#beta_imp = [3,-3,3,-3,-3,3]
beta_unimp = np.zeros(6)
betas = np.concatenate((beta_imp,beta_unimp))

p = 0.7
seeds = [12, 2,4,9,1,11,17,19,20,15,1,3,5,7,91,33,34,35,36,37,54,55,56,57,58]
dp_mat = np.empty([iterations,ncol])
eo_mat = np.empty([iterations,ncol])
acc_mat = np.empty([iterations,ncol])

for i in range(iterations):
  print(i)
  z, X,y = simulate_data(nrow,ncol ,alphas, betas, p, 0, seeds[i], regression=True)
  clf = DecisionTreeRegressor()
  clf.fit(X,y)
  
  #Our approach
  f_forest = fis_tree(clf,X,y,z,0,regression=True,alpha = 0.5)
  f_forest._calculate_fairness_importance_score()
  
  dp_mat[i] = f_forest._fairness_importance_score_dp
  eo_mat[i] = f_forest._fairness_importance_score_eqop
  acc_mat[i] = clf.feature_importances_


#%%
dp_mean = np.mean(dp_mat, axis = 0)
eo_mean = np.mean(eo_mat, axis = 0)
acc_mean = np.mean(acc_mat, axis = 0)


print(f"500 fair: {dp_mean}")
print(f"500 acc: {acc_mean}")
#np.savetxt("/home/camille/FairFIS/results/dp_tree_500.csv", dp_mean, delimiter= ",")

#%%
































#%%
iterations = 10
nrow = 1000
ncol = 12
alphas = [3,3,3,0,0,0,3,3,3,0,0,0,]
np.random.seed(1000)
beta_imp = 3*np.random.uniform(-1,1, size = 6)
beta_unimp = np.zeros(6)
betas = np.concatenate((beta_imp,beta_unimp))

p = 0.7


dp_mat = np.empty([iterations,ncol])
eo_mat = np.empty([iterations,ncol])
acc_mat = np.empty([iterations,ncol])

for i in range(iterations):
  
  z, X,y = simulate_data(nrow,ncol ,alphas, betas, p, 0, seeds[i])
  clf = DecisionTreeRegressor()
  clf.fit(X,y)
  
  #Our approach
  f_forest = fis_tree(clf,X,y,z,0,regression=True)
  f_forest._calculate_fairness_importance_score()
  
  dp_mat[i] = f_forest._fairness_importance_score_dp_root
  eo_mat[i] = f_forest._fairness_importance_score_eqop_root
  acc_mat[i] = clf.feature_importances_


#%%
dp_mean = np.mean(dp_mat, axis = 0)
eo_mean = np.mean(eo_mat, axis = 0)
acc_mean = np.mean(acc_mat, axis = 0)

print(f"1000 fair: {dp_mean}")
print(f"1000 acc: {acc_mean}")
#np.savetxt("/home/camille/FairFIS/results/dp_tree_1000.csv", dp_mean, delimiter= ",")







# %%
