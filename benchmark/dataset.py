#%%
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest
from sklearn.ensemble import RandomForestClassifier



#%%
def select_beta(elements_per_group):
    np.random.seed(1000)
    beta = np.zeros(elements_per_group*4)
    possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        beta[i] = np.random.uniform(7,8)
    for i in range(elements_per_group*2,elements_per_group*3):
        beta[i] = np.random.choice(possibilities)
    #beta[elements_per_group*4] = 20
    return beta
#%%
min_group_01 = 10
max_group_01 = 15

#%%

def toy_4group(elements_per_group, total_samples,z_prob):
    total_features = elements_per_group*4
    z = np.random.binomial(1,z_prob,total_samples)
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            g1[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],1) + np.random.normal(0,1)
            g2[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],1) + np.random.normal(0,1)
        g3[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
        g4[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    beta = select_beta(elements_per_group)
    mu = np.matmul(x,beta) + np.random.normal(0,1,total_samples)
    gama = expit(mu)
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    
    return x,z,y,beta
# %%
elements_per_group = 2
iterations = 1
total_features = elements_per_group * 4 + 1
x, z, y, beta = toy_4group(elements_per_group,2000,0.75)
# %%
