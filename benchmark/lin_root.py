#%%

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest, fis_boosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from FIS import util
import math

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
            value = np.random.uniform(b/7, b/6.5)
        else:
            value = np.random.uniform(-b/6.5,-b/7)
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
iterations = 10
number_of_s = [100,1000,2000]
signals = [0.55,1.25,1.8]
total_features = elements_per_group * 4 + 1
for number_of_samples in number_of_s:
    for b in signals:
        
        beta = select_beta(elements_per_group, b)
        mean_1 = np.random.uniform(min_group_01,max_group_01)
        mean_2 = np.random.uniform(min_group_01,max_group_01)
        occ_dp = np.zeros(total_features - 1)
        occ_eqop = np.zeros(total_features - 1)
        result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','null_dp','null_eqop','feature'])
        
        for i in range (iterations):
            x, z, y, beta, stn = toy_4group(elements_per_group,number_of_samples,0.75,mean_1,mean_2,beta)
            
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
            f_forest = fis_tree(clf,x,y,z,0)
            #f_forest.fit(x,y)
            f_forest._calculate_fairness_importance_score()
            fis_dp, fis_eqop, feature = f_forest.get_root_node_fairness()
            null_dp,null_eqop,null_feature = f_forest.get_null_fairness()

            #######occlusion#########
            
            

            
            result_df = result_df.append({'fis_dp':fis_dp,'fis_eqop':fis_eqop,'null_dp':null_dp,'null_eqop':null_eqop,'feature':feature}, ignore_index=True)

        name = "lin_rootonly"+str(number_of_samples)+"_"+str(elements_per_group)+"_"+str(b)+"1.csv"
        #result_df.to_csv(name)


# %%
