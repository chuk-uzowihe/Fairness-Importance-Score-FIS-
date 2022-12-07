#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_adaboosting, fis_tree, fis_forest, fis_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from FIS import util
import math
from sklearn.base import clone
from copy import deepcopy

#%%
def select_beta(elements_per_group,b):
    np.random.seed(1000)
    beta = np.zeros(elements_per_group*4)
    #possibilities = [7,8,-7,-8]
    for i in range(elements_per_group):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(b/7, b/6.5)
        else:
            value = np.random.uniform(-b/6.5,-b/7)
        beta[i] = value
    for i in range(elements_per_group*2,elements_per_group*3):
        p = np.random.binomial(1,0.5,1)
        if p == 1:
            value = np.random.uniform(2*b/7, 2*b/6.5)
        else:
            value = np.random.uniform(-2*b/6.5,-2*b/7)
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
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y,beta, signal_to_noise

# %%
elements_per_group = 3
iterations = 1
number_of_s = [1000]
signals = [1.8]
total_features = elements_per_group * 4 + 1
for number_of_samples in number_of_s:
    for b in signals:
        
        beta = select_beta(elements_per_group, b)
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        occ_dp = np.zeros(total_features - 1)
        occ_eqop = np.zeros(total_features - 1)
        result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','feature'])
        
        for i in range (iterations):
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,0.75,mean_1,mean_2,beta)
            test_x,test_z,test_y,test_beta,test_stn = toy_4group(elements_per_group,100,0.75,mean_1,mean_2,beta)
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            clf = AdaBoostClassifier(n_estimators=150)
            clf.fit(x,y)
            print(clf.estimator_weights_)
            '''
            prediction = clf.predict_proba(test_x)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)
            clf_f = []
            
            for j in range(4*elements_per_group):
                p = np.zeros((len(test_y),2))
                count = 0
                for i in range (len(clf.estimators_)):
                    if clf.estimators_[i].tree_.feature[0] == j:
                        count+= 1
                        p += (clf.estimators_[i].predict_proba(test_x) * clf.estimator_weights_[i])
                p = p/count
                clf_f.append(prediction-p)
            '''
            
                    #print(feature)






#%%
            #####our approach#########
            f_forest = fis_adaboosting(clf,x,y,z,0)
            #f_forest.fit(x,y)
            f_forest.calculate_fairness_importance_score()
            fis_dp = f_forest._fairness_importance_score_dp
            fis_eqop = f_forest._fairness_importance_score_eqop
            fis_root_dp = f_forest._fairness_importance_score_dp_root
            fis_root_eqop = f_forest._fairness_importance_score_eqop_root

            #######occlusion#########
            
            

            for k in range(total_features-1):
                result_df = result_df.append({'fis_dp':fis_dp[k],'fis_eqop':fis_eqop[k],'dp_root':fis_root_dp[k],'eq_root':fis_root_eqop[k],'stn':stn}, ignore_index=True)

        name = "boosting_nonlin"+str(number_of_samples)+"_"+str(elements_per_group)+"_"+str(b)+"_2.csv"
        #result_df.to_csv(name)


# %%
