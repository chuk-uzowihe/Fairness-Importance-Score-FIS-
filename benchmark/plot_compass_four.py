#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
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
#%%
def single_stn(num_features,iterations,stn1):
    #stn1 = pd.read_csv("result_2_0.1.csv")
    #stn2 = pd.read_csv("result_2_0.7.csv")
    #stn3 = pd.read_csv("result_2_0.95.csv")
    #stn4 = pd.read_csv("result_2_1.2.csv")
    

    
    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    feature_mean = []
    feature_err = []
    for i in range(num_features):
        dp_mean.append(stn1['fis_dp'].iloc[i])
        
        feature_mean.append(stn1['accuracy'].iloc[i])
        
    return dp_mean,feature_mean

# %%
sns.set_context('talk')
fontsize = 20
width = 0.5
#%%
stn1_1 = pd.read_csv("result_tree_compas.csv")
stn1_2 = pd.read_csv("result_rf_compas.csv")
stn1_3 = pd.read_csv("result_boosting_one_compas.csv")
stn1_4 = pd.read_csv("result_boosting_five_compas.csv")

#%%
dp_11, f_11 = single_stn(9,5,stn1_1)
dp_12, f_12 = single_stn(9,5,stn1_2)
dp_13, f_13 = single_stn(9,5,stn1_3)
dp_14, f_14 = single_stn(9,5,stn1_4)
#%%
fig, ax = plt.subplots(2,2,figsize=(12,12))
x_axis = x_axis = np.arange(1,len(dp_11)+1)
ax[0,0].set_ylabel("Importance Score")
ax[1,0].set_ylabel("Importance Score")
fig.supxlabel("Feature")
ax[0,0].set_title("Decision Tree", fontsize = fontsize)
ax[0,0].bar(x_axis - width/2, dp_11,color = 'black', label = 'Fair FIS',width = width)
ax[0,0].bar(x_axis + width/2, f_11,color = 'grey', label = 'FIS',width = width)

ax[0,1].set_title("Random Forest", fontsize = fontsize)
ax[0,1].bar(x_axis - width/2, dp_12,color = 'black', label = 'Fair FIS',width = width)
ax[0,1].bar(x_axis + width/2, f_12,color = 'grey', label = 'FIS',width = width)

ax[1,0].set_title("Gradient Boosting(depth = 1)", fontsize = fontsize)
ax[1,0].bar(x_axis - width/2, dp_13,color = 'black', label = 'Fair FIS',width = width)
ax[1,0].bar(x_axis + width/2, f_13,color = 'grey', label = 'FIS',width = width)

ax[1,1].set_title("Gradient Boosting(depth = 5)", fontsize = fontsize)
ax[1,1].bar(x_axis - width/2, dp_14,color = 'black', label = 'Fair FIS',width = width)
ax[1,1].bar(x_axis + width/2, f_14,color = 'grey', label = 'FIS',width = width)
width = 0.5
fsize = 20
ax[1,0].set_xticks(list(range(1,len(dp_14)+1)))
ax[1,0].set_xticklabels(column_names, fontsize= fontsize, rotation=90)
ax[1,1].set_xticks(list(range(1,len(dp_14)+1)))
ax[1,1].set_xticklabels(column_names, fontsize= fontsize, rotation=90)


plt.legend(loc=2, bbox_to_anchor=(1.05,1.05))
plt.tight_layout()
plt.savefig("compas_all_four.pdf")

#%%

