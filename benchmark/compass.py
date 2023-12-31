#%%
import pandas as pd
import numpy as np
import math
from FIS import fis_tree, fis_forest,fis_boosting
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from FIS import util
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#%%
dataset = pd.read_csv("compas.csv")
data_y = dataset['two_year_recid']
X_df = dataset.drop(['two_year_recid'], axis=1)
X_df = X_df.iloc[:,1:]
X = X_df.to_numpy()
y = data_y.to_numpy()
y = np.where(y != 1, 0, y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

z = X[:,1]
z_test = test_x[:,1]
train_x = np.delete(X, 1, axis = 1)
test_x = np.delete(test_x, 1, axis = 1)

total_features = len(X_df.columns) - 1
column_names = X_df.columns
column_names = column_names.delete(1)
# %%
iterations = 1
result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_std','eq_std','accuracy','accuracy_var'])
dp_fis = {}
eqop_fis = {}
accuracy = {}
[dp_fis.setdefault(i, []) for i in range(total_features)]
[eqop_fis.setdefault(i, []) for i in range(total_features)]
[accuracy.setdefault(i, []) for i in range(total_features)]
for i in range (iterations):
    #x, z, y, beta, stn = toy_4group(elements_per_group,1500,0.75,b)
    
    
    #parameters = {'max_features':[0.5, 0.6, 0.7, 0.8]}
    clf = DecisionTreeClassifier()
    #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)
    clf.fit(train_x,y)
    #####our approach#########
    f_forest = fis_tree(clf,train_x,y,z,0)
    #f_forest.fit(train_x,train_y)
    f_forest._calculate_fairness_importance_score()
    fis_dp = f_forest._fairness_importance_score_dp_root
    fis_eqop = f_forest._fairness_importance_score_eqop_root
    f_importance = clf.feature_importances_
    
    
    for k in range(total_features):
                dp_fis[k].append(fis_dp[k])
                eqop_fis[k].append(fis_eqop[k])
                accuracy[k].append(f_importance[k])
for i in range(total_features):
    result_df = result_df.append({'fis_dp':np.mean(fis_dp[i]),'fis_eqop':np.mean(fis_eqop[i]),'dp_std':np.var(dp_fis[i]),'eq_std':np.var(dp_fis[i]),'accuracy':np.mean(accuracy[i]),'accuracy_var':np.var(accuracy[i])}, ignore_index=True)

name = "result_tree"+"_"+"compas"+".csv"

# %%
result_df.to_csv(name)
# %%
