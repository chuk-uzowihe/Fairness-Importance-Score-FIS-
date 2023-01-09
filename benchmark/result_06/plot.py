#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

#%%
def single_stn(elements_per_group,iterations,stn1):
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
    num_features = 4 * elements_per_group
    for i in range(num_features):
        dp_mean.append(stn1['fis_dp'].iloc[i])
        
        feature_mean.append(stn1['accuracy'].iloc[i])
        
    return dp_mean,feature_mean

# %%
sns.set_context('talk')
fontsize = 20
elements_per_group = 2
width = 0.5
#%%
stn1_1 = pd.read_csv("multiclass_dt.csv")
stn1_2 = pd.read_csv("multiclass_rf.csv")


#%%
dp_11, f_11 = single_stn(2,5,stn1_1)
dp_12, f_12 = single_stn(2,5,stn1_2)

#%%
fig, ax = plt.subplots(2,1,figsize=(6,8),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("Importance Score")
fig.supxlabel("Feature")
width = 0.5
fsize = 20
x_axis = np.arange(1,elements_per_group*4+1)
#color = ['bisque','lightcyan','gold','plum']
color = ['r','g','b','c']
for i in range(4):
    ax[0].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[0].bar(x_axis - width/2,dp_11,width = width, color = 'black')
ax[0].bar(x_axis + width/2,f_11,width = width, color = 'tab:grey')
ax[0].set_title("n = 1000",fontsize = fsize)
ax[0].set_ylabel("Decision Tree", fontsize = fsize)


for i in range(4):
    ax[1].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[1].bar(x_axis - width/2,dp_12,width = width, color = 'black')
ax[1].bar(x_axis + width/2,f_12,width = width, color = 'tab:grey')
ax[1].set_ylabel("Random Forest", fontsize = fsize)


patch1 = mpatches.Patch(color='r', label='Group z & y',alpha = 0.2)
patch2 =mpatches.Patch(color='g', label='Group z',alpha = 0.2)
patch3 =mpatches.Patch(color='b', label='Group y',alpha = 0.2)
patch4 =mpatches.Patch(color='c', label='Group -',alpha = 0.2)
patch5 =mpatches.Patch(color='black', label='Fair FIS')
patch6 =mpatches.Patch(color='tab:grey', label='FIS')

fig.legend(handles = [patch1,patch2,patch3,patch4,patch5,patch6],loc='upper center', bbox_to_anchor=(0.5, -0.01))
plt.savefig("lin_multi_class.pdf")

#%%

