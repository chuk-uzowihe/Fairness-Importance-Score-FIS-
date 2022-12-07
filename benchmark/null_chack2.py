#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from FIS import util

#%%
def select_beta(elements_per_group,b):
    np.random.seed(1000)
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

def toy_4group(elements_per_group, total_samples,z_prob,mean_1,mean_2,beta):
    total_features = elements_per_group*4
    z1_size = int(total_samples * z_prob)
    z1 = np.ones(z1_size)
    z2 = np.zeros(total_samples-z1_size)
    z = np.concatenate((z1,z2))
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            if z[j] == 1:
                g1[i][j] = np.random.normal(mean_1,4)
                g2[i][j] = np.random.normal(mean_1,4)
            else:
                g1[i][j] = np.random.normal(0,4)
                g2[i][j] = np.random.normal(0,4)
            
        g3[i] = np.random.normal(mean_2,4,total_samples)
        g4[i] = np.random.normal(mean_2,4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    
    mu = np.matmul(x,beta) + np.random.normal(0,1,total_samples)
    gama = expit(mu)
    signal_to_noise = np.var(np.matmul(x,beta))
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise

#%%
elements_per_group = 3
iterations = 1
number_of_s = [1000]
signals = [1.25]
total_features = elements_per_group * 4 + 1
ratio = [0.5,0.6,0.7,0.8]
for number_of_samples in number_of_s:
    result_df = pd.DataFrame(columns=['ratio','protected','else'])
    for b in ratio:
        beta = select_beta(elements_per_group, 1.8)
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        occ_dp = np.zeros(total_features - 1)
        occ_eqop = np.zeros(total_features - 1)

        
        
        for i in range (iterations):
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,b,mean_1,mean_2,beta)
            protectedClass = [(x,l) for (x,l) in zip(z, y) 
                    if x == 0] 
            elseClass = [(x,l) for (x,l) in zip(z, y) 
                    if x == 1] 
            p = sum(1 for (x,l) in protectedClass if l == 1)/len(protectedClass)
            q = sum(1 for (x,l) in elseClass  if l == 1)/len(elseClass)
            #######occlusion#########
            result_df = result_df.append({'ratio':b,'protected':p,'else':q}, ignore_index=True)
            

            #for k in range(total_features-1):
            #    result_df = result_df.append({'fis_dp':fis_dp[k],'fis_eqop':fis_eqop[k],'dp_root':fis_root_dp[k],'eq_root':fis_root_eqop[k],'stn':stn}, ignore_index=True)

    name = "expected"+str(number_of_samples)+"_"+str(elements_per_group)+".csv"
    #result_df.to_csv(name)

# %%
