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
import seaborn as sns
import matplotlib.pyplot as plt

#%%
dataset = pd.read_csv("compas.csv")
data_y = dataset['two_year_recid']
X_df = dataset.drop(['two_year_recid'], axis=1)

X = X_df.to_numpy()
y = data_y.to_numpy()
y = np.where(y != 1, 0, y)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

z = train_x[:,1]
z_test = test_x[:,1]
train_x = np.delete(train_x, 1, axis = 1)
test_x = np.delete(test_x, 1, axis = 1)

total_features = len(X_df.columns)
feature_name = X_df.columns
feature_name = np.delete(feature_name, [0,2])
# %%
def single_result(total_features,iterations,stn1):
    #stn1 = pd.read_csv("result_2_0.1.csv")
    #stn2 = pd.read_csv("result_2_0.7.csv")
    #stn3 = pd.read_csv("result_2_0.95.csv")
    #stn4 = pd.read_csv("result_2_1.2.csv")
    
    dp_fis = {}
    eqop_fis = {}
    dp_occ = {}
    eqop_occ = {}
    [dp_fis.setdefault(i, []) for i in range(total_features)]
    [eqop_fis.setdefault(i, []) for i in range(total_features)]
    [dp_occ.setdefault(i, []) for i in range(total_features)]
    [eqop_occ.setdefault(i, []) for i in range(total_features)]

    for i in range(iterations):
        for j in range(1,total_features):
            dp_fis[j].append(stn1['fis_dp'].iloc[i*total_features + j])
            eqop_fis[j].append(stn1['fis_eqop'].iloc[i*total_features + j])
            dp_occ[j].append(stn1['occ_dp'].iloc[i*total_features + j])
            eqop_occ[j].append(stn1['occ_eqop'].iloc[i*total_features + j])
    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    occ_dp_mean = []
    occ_dp_err = []
    occ_eqop_mean = []
    occ_eqop_err = []
    for i in range(1,total_features):
        dp_mean.append(np.mean(dp_fis[i]))
        dp_err.append(np.var(dp_fis[i]))
        eqop_mean.append(np.mean(eqop_fis[i]))
        eqop_err.append(np.var(eqop_fis[i]))
        occ_dp_mean.append(np.mean(dp_occ[i]))
        occ_dp_err.append(np.var(dp_occ[i]))
        occ_eqop_mean.append(np.mean(eqop_occ[i]))
        occ_eqop_err.append(np.var(eqop_occ[i]))
    return dp_mean, dp_err, eqop_mean, eqop_err, occ_dp_mean, occ_dp_err, occ_eqop_mean, occ_eqop_err
# %%

stn1 = pd.read_csv("result_compas.csv")
dp_m1, dp_e1, eq_m1,eq_e1,occd_m1, occd_e1,occe_m1, occe_e1 = single_result(total_features-1,10,stn1)





# %%
sns.set_context('talk')
fontsize = 20
width = 0.5
x_axis = np.arange(1,10,1)

fig, ax = plt.subplots(1,figsize=(25,20))
ax.bar(x_axis,dp_m1,yerr = dp_e1,width = width, color = 'r', label = "FIS")
#ax.bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
#ax.legend()
fig.supylabel("Fairness Importance Score(FIS)", fontsize= fontsize)
fig.supxlabel("Feature", fontsize= fontsize)
#ax.set_xticks([])
ax.set_xticks(list(range(1,10)))
ax.set_title("FIS with EQOP", fontsize= fontsize)
ax.set_xticklabels(feature_name, rotation=45, ha='right', fontsize= fontsize)
plt.savefig("compas_dp.pdf")



#ax.bar(x_axis,occ_mean,yerr = occ_err,width = width, color = 'b', label = "OFS")
# %%
fig, ax = plt.subplots(1,figsize=(25,20))
x_axis = np.arange(1,10,1)

ax.bar(x_axis,eq_m1,yerr = eq_e1,width = width, color = 'r', label = "FIS")
#ax.bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
#ax.legend()
fig.supylabel("Fairness Importance Score(FIS)", fontsize= fontsize)
fig.supxlabel("Feature", fontsize= fontsize)
#ax.set_xticks([])
ax.set_xticks(list(range(1,10)))
ax.set_title("FIS with EQOP", fontsize= fontsize)
ax.set_xticklabels(feature_name, rotation=45, ha='right', fontsize= fontsize)
plt.savefig("compas_eqop.pdf")
# %%
