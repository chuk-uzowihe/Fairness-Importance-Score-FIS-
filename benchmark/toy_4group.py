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
min_group_01 = 5
max_group_01 = 10

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
            g1[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],1) + np.random.normal(0,1)
            g2[i][j] = np.random.normal(np.random.uniform(min_group_01,max_group_01)*z[j],1) + np.random.normal(0,1)
        g3[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
        g4[i] = np.random.normal(np.random.uniform(min_group_01,max_group_01),4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    beta = select_beta(elements_per_group, b)
    mu = np.matmul(x,beta) + np.random.normal(0,1,total_samples)
    gama = expit(mu)
    signal_to_noise = np.var(mu)
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    
    return x,z,y,beta, signal_to_noise


# %%
elements_per_group = 2
iterations = 10
signals = [1,2,3,4]
total_features = elements_per_group * 4 + 1
for b in signals:
    occ_dp = np.zeros(total_features - 1)
    occ_eqop = np.zeros(total_features - 1)
    result_df = pd.DataFrame(columns=['fis_dp','occ_dp','fis_eqop','occ_eqop'])
    total_stn = 0
    for i in range (iterations):
        x, z, y, beta, stn = toy_4group(elements_per_group,1500,0.75,b)
        total_stn += stn
        
        parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
        clf = RandomForestClassifier(n_estimators=100,n_jobs=-2)
        #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

        #####our approach#########
        f_forest = fis_forest(clf,x,y,z,0)
        f_forest.fit(x,y)
        f_forest.calculate_fairness_importance_score()
        fis_dp = f_forest._fairness_importance_score_dp
        fis_eqop = f_forest._fairness_importance_score_eqop
        
        #######occlusion#########
        
        testX,testy, test_z, test_beta, test_stn  = toy_4group(elements_per_group,1000,0.75,2)
    sklearn_tree_all = clf
    sklearn_tree_all.fit(x,y)
    pred_all = sklearn_tree_all.predict(testX)
    testX_with_protected = np.concatenate((testX,np.reshape(test_z,(-1,1))),axis = 1)
    fairness_all_eqop = 1 - util.eqop(testX_with_protected,testy,pred_all,total_features-1,0)
    fairness_all_dp = 1 - util.eqop(testX_with_protected,testy,pred_all,total_features-1,0)
    for j in range (total_features - 1):
        #print(j+1)
        train_data_without_feature = np.delete(x,j,axis=1)
        sklearn_tree= clf
        sklearn_tree.fit(train_data_without_feature,y)
        test_data_without_feature = np.delete(testX,j,axis=1)
        prediction = sklearn_tree.predict(test_data_without_feature)
        test_data_without_feature_with_protected = np.concatenate((test_data_without_feature,np.reshape(test_z,(-1,1))),axis=1)
        occ_dp[j] = fairness_all_dp - (1 - util.DP(test_data_without_feature_with_protected,testy,prediction,total_features-2,0))
        occ_eqop[j] = fairness_all_dp - (1 - util.eqop(test_data_without_feature_with_protected,testy,prediction,total_features-2,0))

        for k in range(total_features-1):
            result_df = result_df.append({'fis_dp':fis_dp[k],'occ_dp':occ_dp[k],'fis_eqop':fis_eqop[k],'occ_eqop':occ_eqop[k]}, ignore_index=True)

        name = "result"+"_"+str(elements_per_group)+"_"+str(math.ceil(stn/iterations))+".csv"
        result_df.to_csv(name)
#%%


# %%
