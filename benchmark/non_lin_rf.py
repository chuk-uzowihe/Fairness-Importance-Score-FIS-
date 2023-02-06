#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV
from FIS import util
import math

#%%
def select_beta(elements_per_group,b):
    beta = np.zeros((elements_per_group - 1)*2)
    #possibilities = [7,8,-7,-8]
    for i in range(len(beta)):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/4)
        if p == 0:
            value = np.random.uniform(-b/4, -b/7)
        beta[i] = value
    #beta[elements_per_group*4] = 20
    return beta
#%%
min_group_01 = 9
max_group_01 = 9.5

min_group_03 = 3
max_group_03 = 3.5

#%%
def additive_func(g1,g2,g3,g4,elements_per_group,total_samples, beta):
    f = np.zeros(total_samples)
    
    for j in range(total_samples):
        f[j] += beta[0]*0.8*(g1[0,j]*g1[1,j]) + beta[1]*g1[2,j] ** 2 + beta[2]*0.8*(g3[0,j]*g3[1,j]) + beta[3]*g3[2,j] ** 2
    return f


#%%

def toy_4group(elements_per_group, total_samples,z_prob,mean,mean3,beta):
    total_features = elements_per_group*4
    z = np.random.binomial(1,z_prob,total_samples)
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            if z[j] == 1:
                if j == elements_per_group - 1:
                    g1[i][j] = np.random.normal(mean,1.8)
                    g2[i][j] = np.random.normal(mean,1.8)
                    g3[i][j] = np.random.normal(mean,1.8)
                    g4[i][j] = np.random.normal(0,1.8)
                else:
                    g1[i][j] = np.random.normal(mean,4)
                    g2[i][j] = np.random.normal(mean,4)
                    g3[i][j] = np.random.normal(mean,4)
                    g4[i][j] = np.random.normal(0,4)
            else:
                if j == elements_per_group - 1:
                    g1[i][j] = np.random.normal(0,1.8)
                    g2[i][j] = np.random.normal(0,1.8)
                    g3[i][j] = np.random.normal(mean,1.8)
                    g4[i][j] = np.random.normal(0,1.8)
                else:
                    g1[i][j] = np.random.normal(0,4)
                    g2[i][j] = np.random.normal(0,4)
                    g3[i][j] = np.random.normal(mean,4)
                    g4[i][j] = np.random.normal(0,4)
            
        #g3[i] = np.random.normal(0,4,total_samples)
        #g4[i] = np.random.normal(0,4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    
    mu = additive_func(g1,g2,g3,g4,elements_per_group,total_samples,beta)
    gama = expit(mu)
    signal_to_noise = np.var(mu)
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y, signal_to_noise

# %%
elements_per_group = 3
iterations = 1
number_of_s = [1000]
signals = [0.15]
total_features = elements_per_group * 4 + 1
for number_of_samples in number_of_s:
    for b in signals:
        
        mean = np.random.uniform(min_group_01,max_group_01)
        mean3 = np.random.uniform(min_group_03,max_group_03)
        beta = select_beta(elements_per_group, b)
        dp_fis = {}
        eqop_fis = {}
        accuracy = {}
        [dp_fis.setdefault(i, []) for i in range(4*elements_per_group)]
        [eqop_fis.setdefault(i, []) for i in range(4*elements_per_group)]
        [accuracy.setdefault(i, []) for i in range(4*elements_per_group)]
        result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_std','eq_std','accuracy','accuracy_var'])
        
        for i in range (iterations):
            x, z, y, stn = toy_4group(elements_per_group,number_of_samples,0.7,mean,mean3,beta)
            
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            clf = RandomForestClassifier(n_estimators=50,n_jobs=-2)
            #clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
            f_forest = fis_forest(clf,x,y,z,0)
            f_forest.fit(x,y)
            f_forest.calculate_fairness_importance_score()
            fis_dp = f_forest._fairness_importance_score_dp_root
            fis_eqop = f_forest._fairness_importance_score_eqop_root
            fis_root_dp = f_forest._fairness_importance_score_dp
            fis_root_eqop = f_forest._fairness_importance_score_eqop

            #######occlusion#########
            
            

            
            feature_importance = f_forest.clf.feature_importances_
            #######occlusion#########
            
            

            for k in range(total_features-1):
                dp_fis[k].append(fis_root_dp[k])
                eqop_fis[k].append(fis_root_eqop[k])
                accuracy[k].append(feature_importance[k])
        for i in range(4*elements_per_group):
            result_df = result_df.append({'fis_dp':np.mean(fis_dp[i]),'fis_eqop':np.mean(fis_eqop[i]),'dp_std':np.var(dp_fis[i]),'eq_std':np.var(dp_fis[i]),'accuracy':np.mean(accuracy[i]),'accuracy_var':np.var(accuracy[i])}, ignore_index=True)

        name = "result_28/nonlin"+str(number_of_samples)+"_"+str(b)+"rf.csv"
        result_df.to_csv(name)


 # %%
