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
total_samples = 1500
#signals = [1,1.6,1.95,2.25]
#signals = [0.1,0.5,0.72,0.88]
#2 signals = [0.1,0.7, 0.95,1.2]
signals = [0.55,1.25,1.8]
total_features = elements_per_group * 4 + 1
for b in signals:
    occ_dp = np.zeros(total_features - 1)
    occ_eqop = np.zeros(total_features - 1)
    result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_root','eq_root','fis_root_dp','fis_root_eqop','feature_root','stn'])
    
    for i in range (iterations):
        x, z, y, beta, stn = toy_4group(elements_per_group,total_samples,0.5,b)
        
        
        #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
        clf = DecisionTreeClassifier(max_depth=8)
        #clf.fit(x,y)
        #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)
        clf.fit(x,y)
        #####our approach#########
        f_forest = fis_tree(clf,x,y,z,0)
        f_forest._calculate_fairness_importance_score()
        fis_dp = f_forest._fairness_importance_score_dp
        fis_eqop = f_forest._fairness_importance_score_eqop
        fis_root_dp = f_forest._fairness_importance_score_dp_root
        fis_root_eqop = f_forest._fairness_importance_score_eqop_root
        #######occlusion#########
        '''
        testX,testy, test_z, test_beta, test_stn  = toy_4group(elements_per_group,1000,0.75,2)
        sklearn_tree_all = clf
        sklearn_tree_all.fit(x,y)
        pred_all = sklearn_tree_all.predict(testX)
        testX_with_protected = np.concatenate((testX,np.reshape(test_z,(-1,1))),axis = 1)
        fairness_all_eqop = 1 - util.eqop(testX_with_protected,testy,pred_all,total_features-1,0)
        fairness_all_dp = 1 - util.eqop(testX_with_protected,testy,pred_all,total_features-1,0)
        for j in range (total_features - 1):
            train_data_without_feature = np.delete(x,j,axis=1)
            sklearn_tree= clf
            sklearn_tree.fit(train_data_without_feature,y)
            test_data_without_feature = np.delete(testX,j,axis=1)
            prediction = sklearn_tree.predict(test_data_without_feature)
            test_data_without_feature_with_protected = np.concatenate((test_data_without_feature,np.reshape(test_z,(-1,1))),axis=1)
            occ_dp[j] = fairness_all_dp - (1 - util.DP(test_data_without_feature_with_protected,testy,prediction,total_features-2,0))
            occ_eqop[j] = fairness_all_dp - (1 - util.eqop(test_data_without_feature_with_protected,testy,prediction,total_features-2,0))
        '''
        for k in range(total_features-1):
            result_df = result_df.append({'fis_dp':fis_dp[k],'fis_eqop':fis_eqop[k],'dp_root':fis_root_dp[k],'eq_root':fis_root_eqop[k],'fis_root_dp':f_forest.dp_at_node[1]-f_forest.dp_at_node[0],'fis_root_eqop':f_forest.eqop_at_node[1]-f_forest.eqop_at_node[0],'feature_root':f_forest.fitted_clf.tree_.feature[0],'stn':stn}, ignore_index=True)

    name = "result"+str(total_samples)+"_"+str(elements_per_group)+"_"+str(b)+"tree_root.csv"
    result_df.to_csv(name)


# %%
