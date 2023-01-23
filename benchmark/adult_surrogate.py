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
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
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

total_features = len(X_df.columns) - 1
column_names = X_df.columns[1:]

# %%
model = Sequential()
model.add(Dense(28, input_dim=98, activation='relu', kernel_initializer="uniform"))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu', kernel_constraint=maxnorm(3), kernel_initializer="uniform"))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu', kernel_initializer="uniform"))
model.add(Dense(1, activation='sigmoid', kernel_initializer="uniform"))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_x, train_y, epochs=100, batch_size=10)
#%%

pred = model.predict(train_x)
test1 = model.predict(test_x)
pred1 = np.zeros(len(pred))
pred_test1 = np.zeros(len(test1))
pred = (pred > 0.5)
for i in range(len(pred)):
    pred1[i] = int(pred[i])
for i in range(len(test1)):
    pred_test1[i] = int(test1[i])
#%%
clf = DecisionTreeClassifier()
clf.fit(train_x,pred1)
pred_test2 = clf.predict(test_x)

#clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

#%%
dp_fis = {}
eqop_fis = {}
accuracy = {}
result_df = pd.DataFrame(columns=['fis_dp','fis_eqop','dp_std','eq_std','accuracy','accuracy_var'])
[dp_fis.setdefault(i, []) for i in range(total_features)]
[eqop_fis.setdefault(i, []) for i in range(total_features)]
[accuracy.setdefault(i, []) for i in range(total_features)]
f_forest = fis_tree(clf,train_x,train_y,z,0)

f_forest._calculate_fairness_importance_score()
fis_dp = f_forest._fairness_importance_score_dp_root
fis_eqop = f_forest._fairness_importance_score_eqop_root
f_importance = f_forest.fitted_clf.feature_importances_
#######occlusion#########

for k in range(total_features):
            dp_fis[k].append(fis_dp[k])
            eqop_fis[k].append(fis_eqop[k])
            accuracy[k].append(f_importance[k])
for i in range(total_features):
    result_df = result_df.append({'fis_dp':np.mean(fis_dp[i]),'fis_eqop':np.mean(fis_eqop[i]),'dp_std':np.var(dp_fis[i]),'eq_std':np.var(dp_fis[i]),'accuracy':np.mean(accuracy[i]),'accuracy_var':np.var(accuracy[i])}, ignore_index=True)

name = "adult/result_surrogate"+"_"+"adult_sex"+".csv"
# %%
# %%


# %%
