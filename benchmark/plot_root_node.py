#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn(stn):
    dp_values = []
    eq_values = []
    feature = []
    for i in range(10):
        dp_values.append(stn.at[i*10,'fis_root_dp'])
        eq_values.append(stn.at[i*10,'fis_root_eqop'])
        feature.append(int(stn.at[i*10,'feature_root']))
    return eq_values,dp_values,feature
#%%
stn1 = pd.read_csv("result500_3_0.55tree_root.csv")
stn2 = pd.read_csv("result500_3_1.25tree_root.csv")
stn3 = pd.read_csv("result500_3_1.8tree_root.csv")

stn1_10 = pd.read_csv("result1000_3_0.55tree_root.csv")
stn2_10 = pd.read_csv("result1000_3_1.25tree_root.csv")
stn3_10 = pd.read_csv("result1000_3_1.8tree_root.csv")

stn1_15 = pd.read_csv("result1500_3_0.55tree_root.csv")
stn2_15 = pd.read_csv("result1500_3_1.25tree_root.csv")
stn3_15 = pd.read_csv("result1500_3_1.8tree_root.csv")


dp1,eq1,feature1 = single_stn(stn1)
dp2,eq2,feature2 = single_stn(stn2)
dp3,eq3,feature3 = single_stn(stn3)

dp1_10,eq1_10,feature1_10 = single_stn(stn1_10)
dp2_10,eq2_10,feature2_10 = single_stn(stn2_10)
dp3_10,eq3_10,feature3_10 = single_stn(stn3_10)

dp1_15,eq1_15,feature1_15 = single_stn(stn1_15)
dp2_15,eq2_15,feature2_15 = single_stn(stn2_15)
dp3_15,eq3_15,feature3_15 = single_stn(stn3_15)

sns.set_context("talk")
width = 0.3
fontsize = 20
fig, ax = plt.subplots(3,3,figsize=(22,18),sharex=True,sharey=True)
x_axis = np.arange(1,11)
fig.supylabel("Fairness Importance Score(FIS)")
fig.supxlabel("Iteration")

ax[0,0].bar(x_axis,dp1,width = width, color = 'r')
for x, y, p in zip(x_axis, dp1, feature1):
   ax[0,0].text(x, y, p)
ax[0,0].set_title("Signal to noise ratio = 1")
ax[0,0].set_ylabel("n = 500", fontsize = fontsize)

ax[0,1].bar(x_axis,dp2,width = width, color = 'r')
for x, y, p in zip(x_axis, dp2, feature1):
   ax[0,1].text(x, y, p)
ax[0,1].set_title("Signal to noise ratio = 5")

ax[0,2].bar(x_axis,dp3,width = width, color = 'r')
for x, y, p in zip(x_axis, dp3, feature1):
   ax[0,2].text(x, y, p)
ax[0,2].set_title("Signal to noise ratio = 10")



ax[1,0].bar(x_axis,dp1_10,width = width, color = 'r')
for x, y, p in zip(x_axis, dp1_10, feature1):
   ax[1,0].text(x, y, p)
ax[1,0].set_title("Signal to noise ratio = 1")
ax[1,0].set_ylabel("n = 1000", fontsize = fontsize)

ax[1,1].bar(x_axis,dp2_10,width = width, color = 'r')
for x, y, p in zip(x_axis, dp2_10, feature1):
   ax[1,1].text(x, y, p)
ax[1,1].set_title("Signal to noise ratio = 5")

ax[1,2].bar(x_axis,dp3_10,width = width, color = 'r')
for x, y, p in zip(x_axis, dp3_10, feature1):
   ax[1,2].text(x, y, p)
ax[1,2].set_title("Signal to noise ratio = 10")


ax[2,0].bar(x_axis,dp1_15,width = width, color = 'r')
for x, y, p in zip(x_axis, dp1_15, feature1):
   ax[2,0].text(x, y, p)
ax[2,0].set_title("Signal to noise ratio = 1")
ax[2,0].set_ylabel("n = 500", fontsize = fontsize)

ax[2,1].bar(x_axis,dp2_15,width = width, color = 'r')
for x, y, p in zip(x_axis, dp2_15, feature1):
   ax[2,1].text(x, y, p)
ax[2,1].set_title("Signal to noise ratio = 5")

ax[2,2].bar(x_axis,dp3_15,width = width, color = 'r')
for x, y, p in zip(x_axis, dp3_15, feature1):
   ax[2,2].text(x, y, p)
ax[2,2].set_title("Signal to noise ratio = 10")

plt.savefig("rootnode_eq.pdf")

# %%
