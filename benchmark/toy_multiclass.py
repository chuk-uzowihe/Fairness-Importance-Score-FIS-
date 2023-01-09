#%%
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from FIS import fis_tree, fis_forest, fis_boosting
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import RandomizedSearchCV
from FIS import util
import math

# %%
k = 3
p = 0.7
size = 1000
non_biased_features = 3
biased_features = 3
elements_per_group = 2
#%%
y = np.random.choice(k,size)
z = np.zeros(size)
# %%
for i in range(len(y)):
    if y[i] <= k/2:
        z[i] = np.random.binomial(1,p,1)
    else:
        z[i] =  np.random.binomial(1,1 - p,1)

# %%
g1 = np.zeros((elements_per_group,size))
g2 = np.zeros((elements_per_group,size))
g3 = np.zeros((elements_per_group,size))
g4 = np.zeros((elements_per_group,size))
mean_k = np.arange(k)

for i in range(elements_per_group):
        for j in range(size):
            if z[j] == 0:
                g2[i][j] = np.random.normal(0,1)
            else:
                g2[i][j] = np.random.normal(1,1)
            g3[i,j] = np.random.normal(2*y[j],1)
            g1[i,j] = np.random.normal(3*y[j] + 3*z[j],1)
            g4[i,j] = np.random.normal(0,1)
x = np.concatenate((np.transpose(g1),np.transpose(g2),np.transpose(g3),np.transpose(g4)),axis = 1)


# %%
result_df = pd.DataFrame(columns=['fis_dp','accuracy'])
clf =  GradientBoostingClassifier(n_estimators=100, max_depth=5, max_features='auto')
clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
f_forest = fis_boosting(clf,x,y,z,0,multiclass = True)
            #f_forest.fit(x,y)
f_forest.calculate_fairness_importance_score()
fis_root_dp = f_forest._fairness_importance_score_dp
fis_root_eqop = f_forest._fairness_importance_score_eqop

feature_importance = clf.feature_importances_


for i in range(4*elements_per_group):
    result_df = result_df.append({'fis_dp':fis_root_dp[i],'accuracy':feature_importance[i]}, ignore_index=True)

name = "result_06/multiclass"+"_"+"boosting_five.csv"
result_df.to_csv(name)



#name = "result_07/rndm_nonlin"+str(number_of_samples)+"_"+str(b)+"rf.csv"
#result_df.to_csv(name)
# %%
