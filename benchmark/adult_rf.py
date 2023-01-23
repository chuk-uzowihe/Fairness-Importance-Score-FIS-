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
dataset = pd.read_csv("adult.csv")
data_y = dataset['income-per-year']
X_df = dataset.drop(['income-per-year'], axis=1)
#X_df = X_df.iloc[: , 1:]
X_df = X_df.iloc[:,1:]
X = X_df.to_numpy()
y = data_y.to_numpy()
y = np.where(y != 1, 0, y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

z = train_x[:,3]
z_test = test_x[:,3]
train_x = np.delete(train_x, 3, axis = 1)
test_x = np.delete(test_x, 3, axis = 1)

total_features = len(X_df.columns) - 1
column_names = X_df.columns
column_names = column_names.delete([3])

# %%
iterations = 10
occ_dp = np.zeros(total_features - 1)
occ_eqop = np.zeros(total_features - 1)

result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_std','eq_std','accuracy','accuracy_var'])
dp_fis = {}
eqop_fis = {}
accuracy = {}
[dp_fis.setdefault(i, []) for i in range(total_features)]
[eqop_fis.setdefault(i, []) for i in range(total_features)]
[accuracy.setdefault(i, []) for i in range(total_features)]
for i in range (iterations):
    #x, z, y, beta, stn = toy_4group(elements_per_group,1500,0.75,b)
    
    
    
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-2)
    #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

    #####our approach#########
    f_forest = fis_forest(clf,train_x,train_y,z,0)
    f_forest.fit(train_x,train_y)
    f_forest.calculate_fairness_importance_score()
    fis_dp = f_forest._fairness_importance_score_dp_root
    fis_eqop = f_forest._fairness_importance_score_eqop_root
    f_importance = f_forest.clf.feature_importances_
    #######occlusion#########
    
    for k in range(total_features):
                dp_fis[k].append(fis_dp[k])
                eqop_fis[k].append(fis_eqop[k])
                accuracy[k].append(f_importance[k])
for i in range(total_features):
    result_df = result_df.append({'fis_dp':np.mean(fis_dp[i]),'fis_eqop':np.mean(fis_eqop[i]),'dp_std':np.var(dp_fis[i]),'eq_std':np.var(dp_fis[i]),'accuracy':np.mean(accuracy[i]),'accuracy_var':np.var(accuracy[i])}, ignore_index=True)

name = "adult/result_rf"+"_"+"adult_sex"+".csv"
#%%
result_df.to_csv(name)
# %%
