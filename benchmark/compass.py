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

#%%
dataset = pd.read_csv("compas.csv")
data_y = dataset['two_year_recid']
X_df = dataset.drop(['two_year_recid'], axis=1)
X_df = X_df.iloc[: , 1:]
X = X_df.to_numpy()
y = data_y.to_numpy()
y = np.where(y != 1, 0, y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

z = train_x[:,1]
z_test = test_x[:,1]
train_x = np.delete(train_x, 1, axis = 1)
test_x = np.delete(test_x, 1, axis = 1)

total_features = len(X_df.columns)

# %%
iterations = 1
occ_dp = np.zeros(total_features - 1)
occ_eqop = np.zeros(total_features - 1)
result_df = pd.DataFrame(columns=['fis_dp','occ_dp','fis_eqop','occ_eqop','feature_importance','stn'])

for i in range (iterations):
    #x, z, y, beta, stn = toy_4group(elements_per_group,1500,0.75,b)
    
    
    parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-2)
    #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

    #####our approach#########
    f_forest = fis_forest(clf,train_x,train_y,z,0)
    f_forest.fit(train_x,train_y)
    f_forest.calculate_fairness_importance_score()
    fis_dp = f_forest._fairness_importance_score_dp
    fis_eqop = f_forest._fairness_importance_score_eqop
    f_importance = f_forest.clf.feature_importances_
    
    #######occlusion#########
    
    #testX,testy, test_z, test_beta, test_stn  = toy_4group(elements_per_group,1000,0.75,2)
    sklearn_tree_all = clf
    sklearn_tree_all.fit(train_x,train_y)
    pred_all = sklearn_tree_all.predict(test_x)
    testX_with_protected = np.concatenate((test_x,np.reshape(z_test,(-1,1))),axis = 1)
    fairness_all_eqop = 1 - util.eqop(testX_with_protected,test_y,pred_all,total_features-1,0)
    fairness_all_dp = 1 - util.eqop(testX_with_protected,test_y,pred_all,total_features-1,0)
    for j in range (total_features - 1):
        train_data_without_feature = np.delete(train_x,j,axis=1)
        sklearn_tree= clf
        sklearn_tree.fit(train_data_without_feature,train_y)
        test_data_without_feature = np.delete(test_x,j,axis=1)
        prediction = sklearn_tree.predict(test_data_without_feature)
        test_data_without_feature_with_protected = np.concatenate((test_data_without_feature,np.reshape(z_test,(-1,1))),axis=1)
        occ_dp[j] = fairness_all_dp - (1 - util.DP(test_data_without_feature_with_protected,test_y,prediction,total_features-2,0))
        occ_eqop[j] = fairness_all_dp - (1 - util.eqop(test_data_without_feature_with_protected,test_y,prediction,total_features-2,0))


    for k in range(total_features-1):
        result_df = result_df.append({'fis_dp':fis_dp[k],'occ_dp':occ_dp[k],'fis_eqop':fis_eqop[k],'occ_eqop':occ_eqop[k],'feature_importance':f_importance[k],'stn':0}, ignore_index=True)

name = "result"+"_"+"compas4"+".csv"
result_df.to_csv(name)
# %%
