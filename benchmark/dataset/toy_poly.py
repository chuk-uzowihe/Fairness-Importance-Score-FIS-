#%%
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from FIS import util

#%%
def select_beta(elements_per_group,b, total_samples):
    beta = np.zeros(total_samples)
    #possibilities = [7,8,-7,-8]
    for i in range(total_samples):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/5)
        else:
            value = np.random.uniform(-b/5,-b/7)
        beta[i] = value
    #beta[elements_per_group*4] = 20
    return beta
#%%
min_group_01 = 1
max_group_01 = 5

#%%

def toy_4group(elements_per_group, total_samples,z_prob,b):
    total_features = elements_per_group*4
    z = np.random.binomial(1,z_prob,total_samples)
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            g1[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],4)
            g2[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],4)
        g3[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
        g4[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)
    phi_x = np.matmul(x,x.T)
    beta = select_beta(elements_per_group, b, total_samples)

    mu = np.matmul(phi_x,beta)
    gama = expit(mu)
    signal_to_noise = np.var(np.matmul(phi_x,beta))
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise
# %%
elements_per_group = 3
iterations = 10
total_samples = 100
x, z, y, beta, stn = toy_4group(elements_per_group,total_samples,0.5,8)
clf = RandomForestClassifier()
#clf.fit(x,y)
#clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)
f_forest = fis_forest(clf,x,y,z,0)
f_forest.fit(x,y)
f_forest.calculate_fairness_importance_score()
# %%
