#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn(total_features,iterations,stn):
    elements_per_group = 3
    dp_values = {}
    eq_values = {}
    dp_root_values = {}
    eq_root_values = {}
    [dp_values.setdefault(i, []) for i in range(total_features)]
    [eq_values.setdefault(i, []) for i in range(total_features)]
    [dp_root_values.setdefault(i, []) for i in range(total_features)]
    [eq_root_values.setdefault(i, []) for i in range(total_features)]
    for i in range(iterations):
        for j in range(total_features):
            dp_values[j].append(stn['fis_dp'].iloc[i*elements_per_group*4 + j])
            eq_values[j].append(stn['fis_eqop'].iloc[i*elements_per_group*4 + j])
            dp_root_values[j].append(stn['dp_root'].iloc[i*elements_per_group*4 + j])
            eq_root_values[j].append(stn['eq_root'].iloc[i*elements_per_group*4 + j])

    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    occ_dp_mean = []
    occ_dp_err = []
    occ_eqop_mean = []
    occ_eqop_err = []
    for i in range(total_features):
        dp_mean.append(np.mean(dp_values[i]))
        dp_err.append(np.var(dp_values[i]))
        eqop_mean.append(np.mean(eq_values[i]))
        eqop_err.append(np.var(eq_values[i]))
        occ_dp_mean.append(np.mean(dp_root_values[i]))
        occ_dp_err.append(np.var(dp_root_values[i]))
        occ_eqop_mean.append(np.mean(eq_root_values[i]))
        occ_eqop_err.append(np.var(eq_root_values[i]))
        
    return  eqop_mean, eqop_err,occ_eqop_mean, occ_eqop_err,dp_mean, dp_err, occ_dp_mean, occ_dp_err
#%%
stn1 = pd.read_csv("rndm1000_3_1.8rf_2.csv")

#%%
dp_m1, dp_e1,dp_root_m1,dp_root_e1, eq_m1,eq_e1, eq_root_m1,eq_root_e1 =single_stn(12,10,stn1)



#%%
#dp_m4, dp_e4, eq_m4,eq_e4,occd_m4, occd_e4,occe_m4, occe_e4 = single_stn_result(elements_per_group,10,stn4) 
elements_per_group = 3
width = 0.3
fontzise = 20
fig, ax = plt.subplots(1,1,figsize=(7,6))
x_axis = np.arange(1,4*elements_per_group + 1)
ax.set_ylabel("Fairness Importance Score(FIS)", fontsize = fontzise)
ax.set_xlabel("Feature", fontsize = fontzise)
ax.bar(np.arange(1,elements_per_group+1),eq_m1[0:elements_per_group],yerr = eq_e1[0:elements_per_group],width = width, color = 'r', label = "Group 1 (z & y)")
ax.bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m1[elements_per_group:elements_per_group*2],yerr = eq_e1[elements_per_group:elements_per_group*2],width = width, color = 'g', label = "Group 2 (z)")
ax.bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m1[elements_per_group*2:elements_per_group*3],yerr = eq_e1[elements_per_group*2:elements_per_group*3],width = width, color = 'b', label = "Group 3 (y)")
ax.bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m1[elements_per_group*3:elements_per_group*4],yerr = eq_e1[elements_per_group*3:elements_per_group*4],width = width, color = '0', label = "Group 4 (-)")
ax.set_title("Signal to noise ratio = 10")


ax.legend(loc = 'lower right')

plt.savefig("dp_one_rf.pdf")

# %%
