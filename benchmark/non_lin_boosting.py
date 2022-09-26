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
min_group_01 = 2
max_group_01 = 3

#%%
def additive_func(g1,g2,g3,g4,elements_per_group,total_samples,a,b):
    f = np.zeros(total_samples)
    for i in range(elements_per_group):
        for j in range(total_samples):
            f[j] += a * math.exp(0.2*g1[i,j]) + b*g3[i,j]
    return f


#%%

def toy_4group(elements_per_group, total_samples,z_prob,mean,a,b):
    total_features = elements_per_group*4
    z = np.random.binomial(1,z_prob,total_samples)
    g1 = np.zeros((elements_per_group,total_samples))
    g2 = np.zeros((elements_per_group,total_samples))
    g3 = np.zeros((elements_per_group,total_samples))
    g4 = np.zeros((elements_per_group,total_samples))
    
    for i in range(elements_per_group):
        for j in range(total_samples):
            if z[j] == 1:
                g1[i][j] = np.random.normal(mean,4)
                g2[i][j] = np.random.normal(mean,4)
            else:
                g1[i][j] = np.random.normal(0,4)
                g2[i][j] = np.random.normal(0,4)
            
        g3[i] = np.random.normal(0,4,total_samples)
        g4[i] = np.random.normal(0,4,total_samples)
    
    
    x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)
    #x = np.concatenate((x,np.reshape(z,(-1,1))),axis = 1)

    
    mu = additive_func(g1,g2,g3,g4,elements_per_group,total_samples,a,b)
    gama = expit(mu + np.random.normal(0,1,total_samples))
    signal_to_noise = np.var(mu)
    y = np.zeros(total_samples)
    for i in range(total_samples):
        y[i] = np.random.binomial(1,gama[i])
    #x = x + np.random.normal(0,1,total_samples) + 
    return x,z,y, signal_to_noise

# %%
elements_per_group = 3
iterations = 1
number_of_s = [100]
signals = [0.15]
total_features = elements_per_group * 4 + 1
for number_of_samples in number_of_s:
    for b in signals:
        
        mean = np.random.uniform(min_group_01,max_group_01)
        occ_dp = np.zeros(total_features - 1)
        occ_eqop = np.zeros(total_features - 1)

        result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_root','eq_root','stn'])
        
        for i in range (iterations):
            x, z, y, stn = toy_4group(elements_per_group,number_of_samples,0.7,mean,b,b)
            
            
            #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
            clf = GradientBoostingClassifier(n_estimators=150, max_depth=1, max_features='auto')
            clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
            f_forest = fis_boosting(clf,x,y,z,0)
            #f_forest.fit(x,y)
            f_forest.calculate_fairness_importance_score_nonstamp()
            fis_dp = f_forest._fairness_importance_score_dp
            fis_eqop = f_forest._fairness_importance_score_eqop
            fis_root_dp = f_forest._fairness_importance_score_dp_root
            fis_root_eqop = f_forest._fairness_importance_score_eqop_root

            #######occlusion#########
            
            

            for k in range(total_features-1):
                result_df = result_df.append({'fis_dp':fis_dp[k],'fis_eqop':fis_eqop[k],'dp_root':fis_root_dp[k],'eq_root':fis_root_eqop[k],'stn':stn}, ignore_index=True)

        name = "boosting_nonlin"+str(number_of_samples)+"_"+str(elements_per_group)+"_"+str(b)+".csv"
        #result_df.to_csv(name)


# %%
