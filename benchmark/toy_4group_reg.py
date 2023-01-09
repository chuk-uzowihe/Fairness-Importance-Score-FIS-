#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from FIS import fis_tree, fis_forest, fis_tree_null
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from FIS import util

#%%
#%%
def select_beta(elements_per_group,b):
    np.random.seed(1000)
    beta = np.zeros(elements_per_group*4)
    #possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = b/((i+1)*7)
        else:
            value = -b/((i+1)*7)
        beta[i] = value
    for i in range(elements_per_group*2,elements_per_group*3):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = 2*b/((i+1)*7)
        else:
            value = -2*b/((i+1)*7)
        beta[i] = value
    #beta[elements_per_group*4] = 20
    return beta
#%%
min_group_01 = 1
max_group_01 = 5

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
    y = mu
    
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise


# %%
elements_per_group = 2
iterations = 1
number_of_s = [1000]
signals = [3]
total_features = elements_per_group * 4 + 1
for number_of_samples in number_of_s:
    for b in signals:
        beta = select_beta(elements_per_group, b)
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        dp_fis = {}
        eqop_fis = {}
        accuracy = {}
        [dp_fis.setdefault(i, []) for i in range(4*elements_per_group)]
        [eqop_fis.setdefault(i, []) for i in range(4*elements_per_group)]
        [accuracy.setdefault(i, []) for i in range(4*elements_per_group)]
        result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_std','eq_std','accuracy','accuracy_var'])
        
        for i in range (iterations):
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,0.75,mean_1,mean_2,beta)
            
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            clf = RandomForestRegressor(n_estimators=100,n_jobs=-2)
            #clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
            f_forest = fis_forest(clf,x,y,z,0,normalize = True,regression=True)
            f_forest.fit(x,y)
            f_forest.calculate_fairness_importance_score()
            #f_forest.get_root_node_fairness()
            fis_dp = f_forest._fairness_importance_score_dp
            fis_eqop = f_forest._fairness_importance_score_eqop
            feature_importance = f_forest.clf.feature_importances_
            #######occlusion#########
            
            

            for k in range(total_features-1):
                dp_fis[k].append(fis_dp[k])
                eqop_fis[k].append(fis_eqop[k])
                accuracy[k].append(feature_importance[k])
        for i in range(4*elements_per_group):
            result_df = result_df.append({'fis_dp':np.mean(fis_dp[i]),'fis_eqop':np.mean(fis_eqop[i]),'dp_std':np.var(dp_fis[i]),'eq_std':np.var(dp_fis[i]),'accuracy':np.mean(accuracy[i]),'accuracy_var':np.var(accuracy[i])}, ignore_index=True)

        name = "result_30/rndm_lin_reg"+str(number_of_samples)+"_"+str(b)+"rf.csv"
        #result_df.to_csv(name)


# %%
