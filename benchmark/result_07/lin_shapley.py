#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest, fis_tree_null
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from FIS import util
from FIS import shapley


#%%
def select_beta(elements_per_group,b):
    #np.random.seed(5)
    beta = np.zeros(elements_per_group*4)
    #possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = b/(0.3*(i+1)*7)
        else:
            value = -b/(0.3*(i+1)*7)
        beta[i] = value
    for i in range(elements_per_group*2,elements_per_group*3):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = 3*b/(0.3*(i+1)*7)
        else:
            value = -3*b/(0.3*(i+1)*7)
        beta[i] = value
    #beta[elements_per_group*4] = 20
    return beta
#%%
min_group_01 = 1
max_group_01 = 5
var = 4

#%%

def toy_4group(elements_per_group, total_samples,z_prob,mean_1,mean_2,beta):
    total_features = elements_per_group*4
    z = np.random.binomial(1,z_prob,total_samples)
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


# %%
elements_per_group = 1
iterations = 1
number_of_s = [100,1000]
signals = [3,6]
total_features = elements_per_group * 4 + 1

for number_of_samples in number_of_s:
    for b in signals:
        
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        beta = select_beta(elements_per_group, b)
        shap = []
        result_df = pd.DataFrame(columns=['shap'])
        for i in range (iterations):
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,0.75,mean_1,mean_2,beta)
            
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            for i in range(4*elements_per_group):
                value = shapley.get_shapley_disc_i(y, x, z, i)
                shap.append(value)
            #######occlusion#########
            
            

            
        for i in range(4*elements_per_group):
            result_df = result_df.append({'shap':shap[i]}, ignore_index=True)

        name = "result_07/rndm_shap"+str(number_of_samples)+"_"+str(b)+"rf.csv"
        #result_df.to_csv(name)


# %%
