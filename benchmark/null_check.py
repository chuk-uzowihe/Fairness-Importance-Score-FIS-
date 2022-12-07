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
    #z = np.random.binomial(1,z_prob,total_samples)
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
number_of_s = [1000,100]
signals = [1.8]
total_features = elements_per_group * 4 + 1
ratio = [0.5]
r = 0.7
for number_of_samples in number_of_s:
    result_df = pd.DataFrame(columns=['ratio','dp','eq'])
    for b in ratio:
        beta = select_beta(elements_per_group, 1.8)
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        occ_dp = np.zeros(total_features - 1)
        occ_eqop = np.zeros(total_features - 1)

        
        
        for i in range (iterations):
            
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,r,mean_1,mean_2,beta)
            x_with_protected = np.concatenate((x,np.reshape(z,(-1,1))),axis=1) 
            left_count = int(number_of_samples*b)
            right_count = number_of_samples - left_count
            total_indeces = np.arange(len(y))
            left_indexs = np.random.choice(len(y),left_count,replace=False)
            right_indexs = [i for i in total_indeces if i not in left_indexs]
            leftX = x_with_protected[left_indexs]
            lefty = y[left_indexs]
            rightX = x_with_protected[right_indexs]
            righty = y[right_indexs]
            print("Sample size",number_of_samples,"left Size",left_count,"Non-protected to total ratio",r)
            fairness_dp = util.fairness(leftX,lefty,rightX,righty,total_features-1,0,2)
            fairness_eq = util.fairness(leftX,lefty,rightX,righty,total_features-1,0,1)
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            print("\n")
            result_df = result_df.append({'ratio':b,'dp':fairness_dp,'eq':fairness_eq}, ignore_index=True)

            #######occlusion#########
            
            

            #for k in range(total_features-1):
            #    result_df = result_df.append({'fis_dp':fis_dp[k],'fis_eqop':fis_eqop[k],'dp_root':fis_root_dp[k],'eq_root':fis_root_eqop[k],'stn':stn}, ignore_index=True)

    name = "null"+str(number_of_samples)+"_"+str(elements_per_group)+str(r)+"1.csv"
    #result_df.to_csv(name)

# %%
