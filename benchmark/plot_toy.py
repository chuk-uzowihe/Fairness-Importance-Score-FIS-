#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn_result(elements_per_group,iterations,stn1):
    #stn1 = pd.read_csv("result_2_0.1.csv")
    #stn2 = pd.read_csv("result_2_0.7.csv")
    #stn3 = pd.read_csv("result_2_0.95.csv")
    #stn4 = pd.read_csv("result_2_1.2.csv")
    
    dp_fis = {}
    eqop_fis = {}
    dp_occ = {}
    eqop_occ = {}
    [dp_fis.setdefault(i, []) for i in range(4*elements_per_group)]
    [eqop_fis.setdefault(i, []) for i in range(4*elements_per_group)]
    [dp_occ.setdefault(i, []) for i in range(4*elements_per_group)]
    [eqop_occ.setdefault(i, []) for i in range(4*elements_per_group)]

    for i in range(iterations):
        for j in range(4*elements_per_group):
            dp_fis[j].append(stn1['fis_dp'].iloc[i*elements_per_group*4 + j])
            eqop_fis[j].append(stn1['fis_eqop'].iloc[i*elements_per_group*4 + j])
            dp_occ[j].append(stn1['occ_dp'].iloc[i*elements_per_group*4 + j])
            eqop_occ[j].append(stn1['occ_eqop'].iloc[i*elements_per_group*4 + j])
    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    occ_dp_mean = []
    occ_dp_err = []
    occ_eqop_mean = []
    occ_eqop_err = []
    for i in range(4*elements_per_group):
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
sns.set_context('talk')
fontsize = 15
elements_per_group = 3
width = 0.3
stn1 = pd.read_csv("result_3_0.1.csv")
stn2 = pd.read_csv("result_3_0.5.csv")
stn3 = pd.read_csv("result_3_0.72.csv")
stn4 = pd.read_csv("result_3_0.88.csv")
dp_m1, dp_e1, eq_m1,eq_e1,occd_m1, occd_e1,occe_m1, occe_e1 = single_stn_result(2,10,stn1)
dp_m2, dp_e2, eq_m2,eq_e2,occd_m2, occd_e2,occe_m2, occe_e2 = single_stn_result(2,10,stn2)
dp_m3, dp_e3, eq_m3,eq_e3,occd_m3, occd_e3,occe_m3, occe_e3 = single_stn_result(2,10,stn3)
dp_m4, dp_e4, eq_m4,eq_e4,occd_m4, occd_e4,occe_m4, occe_e4 = single_stn_result(2,10,stn4) 
fig, ax = plt.subplots(2,2,figsize=(16,10),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("Importance Score")
fig.supxlabel("Feature")
ax[0,0].bar(x_axis- width,dp_m1,yerr = dp_e1,width = width, color = 'r', label = "FIS")
ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[0,0].set_title("Signal to noise ratio = 4")

ax[0,1].bar(x_axis- width,dp_m2,yerr = dp_e2,width = width, color = 'r', label = "FIS")
ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[0,1].set_title("Signal to noise ratio = 3")
ax[0,1].legend()

ax[1,0].bar(x_axis- width,dp_m3,yerr = dp_e3,width = width, color = 'r', label = "FIS")
ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[1,0].set_title("Signal to noise ratio = 2")

ax[1,1].bar(x_axis- width,dp_m4,yerr = dp_e4,width = width, color = 'r', label = "FIS")
ax[1,1].bar(x_axis,occd_m4,yerr = occd_e4,width = width, color = 'b', label = "OFS")
ax[1,1].set_title("Signal to noise ratio = 1")



plt.savefig("toy_stn_dp_3.pdf")

# %%
fig, ax = plt.subplots(2,2,figsize=(16,10),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("Importance Score")
fig.supxlabel("Feature")
ax[0,0].bar(x_axis- width,eq_m1,yerr = eq_e1,width = width, color = 'r', label = "FIS")
ax[0,0].bar(x_axis,occe_m1,yerr = occe_e1,width = width, color = 'b', label = "OFS")
ax[0,0].set_title("Signal to noise ratio = 4")

ax[0,1].bar(x_axis- width,eq_m2,yerr = eq_e2,width = width, color = 'r', label = "FIS")
ax[0,1].bar(x_axis,occe_m2,yerr = occe_e2,width = width, color = 'b', label = "OFS")
ax[0,1].set_title("Signal to noise ratio = 3")
ax[0,1].legend()

ax[1,0].bar(x_axis- width,eq_m3,yerr = eq_e3,width = width, color = 'r', label = "FIS")
ax[1,0].bar(x_axis,occe_m3,yerr = occe_e3,width = width, color = 'b', label = "OFS")
ax[1,0].set_title("Signal to noise ratio = 2")

ax[1,1].bar(x_axis- width,eq_m4,yerr = eq_e4,width = width, color = 'r', label = "FIS")
ax[1,1].bar(x_axis,occe_m4,yerr = occe_e4,width = width, color = 'b', label = "OFS")
ax[1,1].set_title("Signal to noise ratio = 1")



plt.savefig("toy_stn_eq_3.pdf")

# %%
