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
    elements_per_group = 2

    
    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    feature_mean = []
    feature_err = []
    num_features = 4 * elements_per_group
    for i in range(num_features):
        dp_mean.append(stn1['fis_dp'].iloc[i])
        dp_err.append(stn1['dp_std'].iloc[i])
        eqop_mean.append(stn1['fis_eqop'].iloc[i])
        eqop_err.append(stn1['eq_std'].iloc[i])
        feature_mean.append(stn1['accuracy'].iloc[i])
        feature_err.append(stn1['accuracy_var'].iloc[i])
    return dp_mean, eqop_mean,feature_mean

# %%
sns.set_context('talk')
fontsize = 20
elements_per_group = 2
width = 0.5
#%%
stn1_1 = pd.read_csv("rndm_lin_reg100_lin_3dt.csv")
stn1_2 = pd.read_csv("rndm_lin_reg100_lin_6dt.csv")
stn1_3 = pd.read_csv("rndm_lin_reg1000_lin_3dt.csv")
stn1_4 = pd.read_csv("rndm_lin_reg1000_lin_6dt.csv")

stn2_1 = pd.read_csv("rndm_lin_reg100_3rf.csv")
stn2_2 = pd.read_csv("rndm_lin_reg100_6rf.csv")
stn2_3 = pd.read_csv("rndm_lin_reg1000_3rf.csv")
stn2_4 = pd.read_csv("rndm_lin_reg1000_6rf.csv")


stn3_1 = pd.read_csv("boosting_lin_reg100_3one.csv")
stn3_2 = pd.read_csv("boosting_lin100_4.5one.csv")
stn3_3 = pd.read_csv("boosting_lin1000_1.5one.csv")
stn3_4 = pd.read_csv("boosting_lin1000_4.5one.csv")

stn4_1 = pd.read_csv("boosting_lin100_1.5five.csv")
stn4_2 = pd.read_csv("boosting_lin100_4.5five.csv")
stn4_3 = pd.read_csv("boosting_lin1000_1.5five.csv")
stn4_4 = pd.read_csv("boosting_lin1000_4.5five.csv")

#%%
dp_11, eq_11, f_11 = single_stn(2,5,stn1_1)
dp_12, eq_12, f_12 = single_stn(2,5,stn1_2)
dp_13, eq_13, f_13 = single_stn(2,5,stn1_3)
dp_14, eq_14, f_14 = single_stn(2,5,stn1_4)


dp_21, eq_21, f_21 = single_stn(2,5,stn2_1)
dp_22, eq_22, f_22 = single_stn(2,5,stn2_2)
dp_23, eq_23, f_23 = single_stn(2,5,stn2_3)
dp_24, eq_24, f_24 = single_stn(2,5,stn2_4)

dp_31, eq_31, f_31 = single_stn(2,5,stn3_1)
dp_32, eq_32, f_32 = single_stn(2,5,stn3_2)
dp_33, eq_33, f_33 = single_stn(2,5,stn3_3)
dp_34, eq_34, f_34 = single_stn(2,5,stn3_4)

dp_41, eq_41, f_41 = single_stn(2,5,stn4_1)
dp_42, eq_42, f_42 = single_stn(2,5,stn4_2)
dp_43, eq_43, f_43 = single_stn(2,5,stn4_3)
dp_44, eq_44, f_44 = single_stn(2,5,stn4_4)
#%%
fig, ax = plt.subplots(4,4,figsize=(22,18),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("Importance Score")
fig.supxlabel("Feature")
width = 0.5
fsize = 20
x_axis = np.arange(1,elements_per_group*4+1)
#color = ['bisque','lightcyan','gold','plum']
color = ['r','g','b','c']
for i in range(4):
    ax[0,0].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[0,0].bar(x_axis - width/2,dp_11,width = width, color = 'black')
ax[0,0].bar(x_axis + width/2,f_11,width = width, color = 'tab:grey')
ax[0,0].set_title("SNR = 1, n = 100",fontsize = fsize)
ax[0,0].set_ylabel("Decision Tree", fontsize = fsize)
for i in range(4):
    ax[0,1].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[0,1].bar(x_axis - width/2,dp_12,width = width, color = 'black')
ax[0,1].bar(x_axis + width/2,f_12,width = width, color = 'tab:grey')
ax[0,1].set_title("SNR = 10, n = 100",fontsize = fsize)
for i in range(4):
    ax[0,2].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[0,2].bar(x_axis - width/2,dp_13,width = width, color = 'black')
ax[0,2].bar(x_axis + width/2,f_13,width = width, color = 'tab:grey')
ax[0,2].set_title("SNR = 1, n = 1000",fontsize = fsize)
for i in range(4):
    ax[0,3].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[0,3].bar(x_axis - width/2,dp_14,width = width, color = 'black')
ax[0,3].bar(x_axis + width/2,f_14,width = width, color = 'tab:grey')
ax[0,3].set_title("SNR = 10, n = 1000",fontsize = fsize)

for i in range(4):
    ax[1,0].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[1,0].bar(x_axis - width/2,dp_21,width = width, color = 'black')
ax[1,0].bar(x_axis + width/2,f_21,width = width, color = 'tab:grey')
ax[1,0].set_ylabel("Random Forest", fontsize = fsize)
for i in range(4):
    ax[1,1].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[1,1].bar(x_axis - width/2,dp_22,width = width, color = 'black')
ax[1,1].bar(x_axis + width/2,f_22,width = width, color = 'tab:grey')

for i in range(4):
    ax[1,2].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[1,2].bar(x_axis - width/2,dp_23,width = width, color = 'black')
ax[1,2].bar(x_axis + width/2,f_23,width = width, color = 'tab:grey')

for i in range(4):
    ax[1,3].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[1,3].bar(x_axis - width/2,dp_24,width = width, color = 'black')
ax[1,3].bar(x_axis + width/2,f_24,width = width, color = 'tab:grey')


for i in range(4):
    ax[2,0].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[2,0].bar(x_axis - width/2,dp_31,width = width, color = 'black')
ax[2,0].bar(x_axis + width/2,f_31,width = width, color = 'tab:grey')
ax[2,0].set_ylabel("Boosting(depth = 1)", fontsize = fsize)
for i in range(4):
    ax[2,1].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[2,1].bar(x_axis - width/2,dp_32,width = width, color = 'black')
ax[2,1].bar(x_axis + width/2,f_32,width = width, color = 'tab:grey')

for i in range(4):
    ax[2,2].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[2,2].bar(x_axis - width/2,dp_33,width = width, color = 'black')
ax[2,2].bar(x_axis + width/2,f_33,width = width, color = 'tab:grey')

for i in range(4):
    ax[2,3].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[2,3].bar(x_axis - width/2,dp_34,width = width, color = 'black')
ax[2,3].bar(x_axis + width/2,f_34,width = width, color = 'tab:grey')

for i in range(4):
    ax[3,0].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[3,0].bar(x_axis - width/2,dp_41,width = width, color = 'black')
ax[3,0].bar(x_axis + width/2,f_41,width = width, color = 'tab:grey')
ax[3,0].set_ylabel("Boosting(depth = 5)", fontsize = fsize)
for i in range(4):
    ax[3,1].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[3,1].bar(x_axis - width/2,dp_42,width = width, color = 'black')
ax[3,1].bar(x_axis + width/2,f_42,width = width, color = 'tab:grey')

for i in range(4):
    ax[3,2].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[3,2].bar(x_axis - width/2,dp_43,width = width, color = 'black')
ax[3,2].bar(x_axis + width/2,f_43,width = width, color = 'tab:grey')

for i in range(4):
    ax[3,3].axvspan(i*elements_per_group+1 - width,(i+1)*elements_per_group + width,facecolor=color[i],alpha = 0.2)
ax[3,3].bar(x_axis - width/2,dp_44,width = width, color = 'black')
ax[3,3].bar(x_axis + width/2,f_44,width = width, color = 'tab:grey')

patch1 = mpatches.Patch(color='r', label='Group z & y',alpha = 0.2)
patch2 =mpatches.Patch(color='g', label='Group z',alpha = 0.2)
patch3 =mpatches.Patch(color='b', label='Group y',alpha = 0.2)
patch4 =mpatches.Patch(color='c', label='Group -',alpha = 0.2)
patch5 =mpatches.Patch(color='black', label='Fair FIS')
patch6 =mpatches.Patch(color='tab:grey', label='FIS')

fig.legend(handles = [patch1,patch2,patch3,patch4,patch5,patch6],loc='upper center', bbox_to_anchor=(0.5, -0.01))
plt.savefig("lin_reg.pdf")

#%%





#%%
stn1_1 = pd.read_csv("rndm100_lin_1.5dt.csv")
stn1_2 = pd.read_csv("rndm100_lin_4.5dt.csv")
stn1_3 = pd.read_csv("rndm1000_lin_1.5dt.csv")
stn1_4 = pd.read_csv("rndm1000_lin_4.5dt.csv")

stn2_1 = pd.read_csv("rndm_lin100_1.5rf.csv")
stn2_2 = pd.read_csv("rndm_lin100_4.5rf.csv")
stn2_3 = pd.read_csv("rndm_lin1000_1.5rf.csv")
stn2_4 = pd.read_csv("rndm_lin1000_4.5rf.csv")


stn3_1 = pd.read_csv("boosting_lin100_1.5one.csv")
stn3_2 = pd.read_csv("boosting_lin100_4.5one.csv")
stn3_3 = pd.read_csv("boosting_lin1000_1.5one.csv")
stn3_4 = pd.read_csv("boosting_lin1000_4.5one.csv")

stn4_1 = pd.read_csv("boosting_lin100_1.5five.csv")
stn4_2 = pd.read_csv("boosting_lin100_4.5five.csv")
stn4_3 = pd.read_csv("boosting_lin1000_1.5five.csv")
stn4_4 = pd.read_csv("boosting_lin1000_4.5five.csv")




stn1_1 = pd.read_csv("rndm_nonlin100_0.6dt_2.csv")
stn1_2 = pd.read_csv("rndm_nonlin100_0.9dt_2.csv")
stn1_3 = pd.read_csv("rndm_nonlin1000_0.6dt_2.csv")
stn1_4 = pd.read_csv("rndm_nonlin1000_0.9dt_2.csv")

stn2_1 = pd.read_csv("rndm_nonlin100_0.6rf.csv")
stn2_2 = pd.read_csv("rndm_nonlin100_0.9rf.csv")
stn2_3 = pd.read_csv("rndm_nonlin1000_0.6rf.csv")
stn2_4 = pd.read_csv("rndm_nonlin1000_0.9rf.csv")


stn3_1 = pd.read_csv("boosting_nonlin100_0.6one.csv")
stn3_2 = pd.read_csv("boosting_nonlin100_0.9one.csv")
stn3_3 = pd.read_csv("boosting_nonlin1000_0.6one.csv")
stn3_4 = pd.read_csv("boosting_nonlin1000_0.9one.csv")

stn4_1 = pd.read_csv("boosting_nonlin100_0.6five.csv")
stn4_2 = pd.read_csv("boosting_nonlin100_0.9five.csv")
stn4_3 = pd.read_csv("boosting_nonlin1000_0.6five.csv")
stn4_4 = pd.read_csv("boosting_nonlin1000_0.9five.csv")