#%%
import pandas as pd
import numpy as np
import math
from FIS import fis_tree, fis_forest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from FIS import util
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

#%%
dataset = pd.read_csv("adult.csv")
data_y = dataset['income-per-year']
X_df = dataset.drop(['income-per-year'], axis=1)

X = X_df.to_numpy()
y = data_y.to_numpy()
y = np.where(y != 1, 0, y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

z = train_x[:,3]
z_test = test_x[:,3]
train_x = np.delete(train_x, 3, axis = 1)
test_x = np.delete(test_x, 3, axis = 1)

total_features = len(X_df.columns)

# %%
iterations = 1
occ_dp = np.zeros(total_features - 1)
occ_eqop = np.zeros(total_features - 1)
result_df = pd.DataFrame(columns=['fis_dp','acc','stn'])

for i in range (iterations):
    #x, z, y, beta, stn = toy_4group(elements_per_group,1500,0.75,b)
    
    
    parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-2)
    #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

    #####our approach#########
    f_forest = fis_forest(clf,train_x,train_y,z,0)
    f_forest.fit(train_x,train_y)
    f_forest.calculate_fairness_importance_score()
    fis_dp = f_forest._fairness_importance_score_dp_root
    fis_eqop = f_forest._fairness_importance_score_eqop
    feature_importance = f_forest.clf.feature_importances_
    #######occlusion#########
    
    #testX,testy, test_z, test_beta, test_stn  = toy_4group(elements_per_group,1000,0.75,2)
    

    for k in range(total_features-1):
        result_df = result_df.append({'fis_dp':fis_dp[k],'acc':feature_importance[i]}, ignore_index=True)

name = "result_rf"+"_"+"adult_sex"+".csv"
result_df.to_csv(name)
# %%
