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
def select_beta(elements_per_group,b):
    beta = np.zeros(elements_per_group*4)
    #possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/5)
        else:
            value = np.random.uniform(-b/5,-b/7)
        beta[i] = value
    for i in range(elements_per_group*2,elements_per_group*3):
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

    beta = select_beta(elements_per_group, b)
    mu = np.matmul(x,beta)
    gama = expit(mu)
    signal_to_noise = np.var(np.matmul(x,beta))
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise






# %%
elements_per_group = 3
iterations = 10
total_samples = 1000
#signals = [1,1.6,1.95,2.25]
#signals = [0.1,0.5,0.72,0.88]
#2 signals = [0.1,0.7, 0.95,1.2]
signals = [0.55,1.25,1.8]

total_features = elements_per_group * 4 + 1
for b in signals:
    result_df = pd.DataFrame(columns=['dp','eqop','dp_root','eqop_root','stn'])
    x, z, y, beta, stn = toy_4group(elements_per_group,total_samples,0.5,b)
    
    for i in range(iterations):
    
    #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
        clf = DecisionTreeClassifier()
        #clf.fit(x,y)
        #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)
        clf.fit(x,y)
        f_forest = fis_tree(clf,x,y,z,0)
        f_forest._calculate_fairness_importance_score()
        fis_dp = f_forest._fairness_importance_score_dp
        fis_eqop = f_forest._fairness_importance_score_eqop
        fis_root_dp = f_forest._fairness_importance_score_dp_root
        fis_root_eqop = f_forest._fairness_importance_score_eqop_root
        result_df = result_df.append({'dp':fis_dp, 'eqop':fis_eqop,'dp_root':fis_root_dp, 'eqop_root':fis_root_eqop,'stn':stn}, ignore_index=True)
    name = "bern"+"dp_eq"+str(total_samples)+"_"+str(b)+".csv"
    result_df.to_csv(name)

# %%

# %%
