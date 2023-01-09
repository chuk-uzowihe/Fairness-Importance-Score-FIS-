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
model.fit(train_x, train_y, epochs=10, batch_size=10)
pred = model.predict(train_x)
pred = (pred > 0.5)
clf = DecisionTreeClassifier()
            #clf.fit(x,y)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
clf.fit(train_x,pred)
            #clf = RandomizedSearchCV(estimator = rf, param_distributions = parameters)

            #####our approach#########
f_forest = fis_tree(clf,train_x,pred,z,0)
            
f_forest._calculate_fairness_importance_score()
fis_dp = f_forest._fairness_importance_score_dp
# %%
feature_imp = clf.feature_importances_
# %%
indexes = np.nonzero(fis_dp)
dps = fis_dp[indexes]
imps = feature_imp[indexes]
names = X_df.columns[indexes]
# %%
sns.set_context('talk')
fontsize = 20
width = 0.5
x_axis = np.arange(1,len(dps)+1)
fig, ax = plt.subplots(1,1, figsize=(10, 8),sharex=True)
ax.bar(x_axis - width/2, dps,color = 'black', label = 'Fair FIS',width = width)
ax.bar(x_axis + width/2, imps,color = 'grey', label = 'FIS',width = width)
ax.set_xticklabels(names, fontsize= fontsize, rotation=90)
ax.set_xticks(list(range(1,len(dps)+1)))
ax.set_ylabel("Importance Score", fontsize = fontsize)
ax.set_ylabel("Feature", fontsize = fontsize)

fig.legend()
plt.savefig("output_age.pdf")
plt.show()
# %%
