#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn(elements_per_group,iterations,stn1):
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
            dp_occ[j].append(stn1['dp_root'].iloc[i*elements_per_group*4 + j])
            eqop_occ[j].append(stn1['eq_root'].iloc[i*elements_per_group*4 + j])
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
    return dp_mean, dp_err, occ_dp_mean, occ_dp_err,eqop_mean,eqop_err,occ_eqop_mean, occ_eqop_err

# %%
sns.set_context('talk')
fontsize = 20
elements_per_group = 3
width = 0.5
#%%
stn1 = pd.read_csv("boosting100_3_0.55binom1.csv")
stn2 = pd.read_csv("boosting100_3_1.25binom1.csv")
stn3 = pd.read_csv("boosting100_3_1.8binom1.csv")

stn1_10 = pd.read_csv("boosting500_3_0.55binom1.csv")
stn2_10 = pd.read_csv("boosting500_3_1.25binom1.csv")
stn3_10 = pd.read_csv("boosting500_3_1.8binom1.csv")

stn1_15 = pd.read_csv("boosting1000_3_0.55binom1.csv")
stn2_15 = pd.read_csv("boosting1000_3_1.25binom1.csv")
stn3_15 = pd.read_csv("boosting1000_3_1.8binom1.csv")
#%%
dp_m1, dp_e1,dp_root_m1,dp_root_e1, eq_m1,eq_e1, eq_root_m1,eq_root_e1 =single_stn(elements_per_group,10,stn1)
dp_m2, dp_e2,dp_root_m2,dp_root_e2, eq_m2,eq_e2, eq_root_m2,eq_root_e2 = single_stn(elements_per_group,10,stn2)
dp_m3, dp_e3,dp_root_m3,dp_root_e3, eq_m3,eq_e3, eq_root_m3,eq_root_e3 = single_stn(elements_per_group,10,stn3)


dp_m1_10, dp_e1_10,dp_root_m1_10,dp_root_e1_10, eq_m1_10,eq_e1_10, eq_root_m1_10,eq_root_e1_10 =single_stn(elements_per_group,10,stn1_10)
dp_m2_10, dp_e2_10,dp_root_m2_10,dp_root_e2_10, eq_m2_10,eq_e2_10, eq_root_m2_10,eq_root_e2_10 = single_stn(elements_per_group,10,stn2_10)
dp_m3_10, dp_e3_10,dp_root_m3_10,dp_root_e3_10, eq_m3_10,eq_e3_10, eq_root_m3_10,eq_root_e3_10 = single_stn(elements_per_group,10,stn3_10)


dp_m1_15, dp_e1_15,dp_root_m1_15,dp_root_e1_15, eq_m1_15,eq_e1_15, eq_root_m1_15,eq_root_e1_15 =single_stn(elements_per_group,10,stn1_15)
dp_m2_15, dp_e2_15,dp_root_m2_15,dp_root_e2_15, eq_m2_15,eq_e2_15, eq_root_m2_15,eq_root_e2_15 = single_stn(elements_per_group,10,stn2_15)
dp_m3_15, dp_e3_15,dp_root_m3_15,dp_root_e3_15, eq_m3_15,eq_e3_15, eq_root_m3_15,eq_root_e3_15 = single_stn(elements_per_group,10,stn3_15)

fig, ax = plt.subplots(3,3,figsize=(22,18),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("FIS")
fig.supxlabel("Feature")

ax[0,0].bar(np.arange(1,elements_per_group+1),eq_m1[0:elements_per_group],yerr = eq_e1[0:elements_per_group],width = width, color = 'r', label = "Group 1 (z & y)")
ax[0,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m1[elements_per_group:elements_per_group*2],yerr = eq_e1[elements_per_group:elements_per_group*2],width = width, color = 'g', label = "Group 2 (z)")
ax[0,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m1[elements_per_group*2:elements_per_group*3],yerr = eq_e1[elements_per_group*2:elements_per_group*3],width = width, color = 'b', label = "Group 3 (y)")
ax[0,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m1[elements_per_group*3:elements_per_group*4],yerr = eq_e1[elements_per_group*3:elements_per_group*4],width = width, color = '0', label = "Group 4 (-)")
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[0,0].set_title("Signal to noise ratio = 1")
ax[0,0].set_ylabel("n = 100", fontsize = 20)

ax[0,1].bar(np.arange(1,elements_per_group+1),eq_m2[0:elements_per_group],yerr = eq_e2[0:elements_per_group],width = width, color = 'r')
ax[0,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m2[elements_per_group:elements_per_group*2],yerr = eq_e2[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[0,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m2[elements_per_group*2:elements_per_group*3],yerr = eq_e2[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[0,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m2[elements_per_group*3:elements_per_group*4],yerr = eq_e2[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[0,1].set_title("Signal to noise ratio = 5")
#ax[0,1].legend()

ax[0,2].bar(np.arange(1,elements_per_group+1),eq_m3[0:elements_per_group],yerr = eq_e3[0:elements_per_group],width = width, color = 'r')
ax[0,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m3[elements_per_group:elements_per_group*2],yerr = eq_e3[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[0,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m3[elements_per_group*2:elements_per_group*3],yerr = eq_e3[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[0,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m3[elements_per_group*3:elements_per_group*4],yerr = eq_e3[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[0,2].set_title("Signal to noise ratio = 10")
#ax[0].set_title("sample size = 500")


ax[1,0].bar(np.arange(1,elements_per_group+1),eq_m1_10[0:elements_per_group],yerr = eq_e1_10[0:elements_per_group],width = width, color = 'r')
ax[1,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m1_10[elements_per_group:elements_per_group*2],yerr = eq_e1_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m1_10[elements_per_group*2:elements_per_group*3],yerr = eq_e1_10[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[1,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m1_10[elements_per_group*3:elements_per_group*4],yerr = eq_e1_10[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[1,0].set_title("Signal to noise ratio = 1")
ax[1,0].set_ylabel("n = 500", fontsize = 20)

ax[1,1].bar(np.arange(1,elements_per_group+1),eq_m2_10[0:elements_per_group],yerr = eq_e2_10[0:elements_per_group],width = width, color = 'r')
ax[1,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m2_10[elements_per_group:elements_per_group*2],yerr = eq_e2_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m2_10[elements_per_group*2:elements_per_group*3],yerr = eq_e2_10[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[1,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m2_10[elements_per_group*3:elements_per_group*4],yerr = eq_e2_10[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[1,1].set_title("Signal to noise ratio = 5")
#ax[0,1].legend()

ax[1,2].bar(np.arange(1,elements_per_group+1),eq_m3_10[0:elements_per_group],yerr = eq_e3_10[0:elements_per_group],width = width, color = 'r')
ax[1,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m3_10[elements_per_group:elements_per_group*2],yerr = eq_e3_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m3_10[elements_per_group*2:elements_per_group*3],yerr = eq_e3_10[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[1,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m3_10[elements_per_group*3:elements_per_group*4],yerr = eq_e3_10[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[1,2].set_title("Signal to noise ratio = 10")

#ax[1].set_title("sample size = 500")



ax[2,0].bar(np.arange(1,elements_per_group+1),eq_m1_15[0:elements_per_group],yerr = eq_e1_15[0:elements_per_group],width = width, color = 'r')
ax[2,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m1_15[elements_per_group:elements_per_group*2],yerr = eq_e1_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m1_15[elements_per_group*2:elements_per_group*3],yerr = eq_e1_15[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[2,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m1_15[elements_per_group*3:elements_per_group*4],yerr = eq_e1_15[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[2,0].set_title("Signal to noise ratio = 1")
ax[2,0].set_ylabel("n = 1000", fontsize = 20)
ax[2,0].set_xticks(np.arange(2,18,2))

ax[2,1].bar(np.arange(1,elements_per_group+1),eq_m2_15[0:elements_per_group],yerr = eq_e2_15[0:elements_per_group],width = width, color = 'r')
ax[2,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m2_15[elements_per_group:elements_per_group*2],yerr = eq_e2_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m2_15[elements_per_group*2:elements_per_group*3],yerr = eq_e2_15[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[2,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m2_15[elements_per_group*3:elements_per_group*4],yerr = eq_e2_15[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[2,1].set_title("Signal to noise ratio = 5")
#ax[0,1].legend()

ax[2,2].bar(np.arange(1,elements_per_group+1),eq_m3_15[0:elements_per_group],yerr = eq_e3_15[0:elements_per_group],width = width, color = 'r')
ax[2,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1),eq_m3_15[elements_per_group:elements_per_group*2],yerr = eq_e3_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1),eq_m3_15[elements_per_group*2:elements_per_group*3],yerr = eq_e3_15[elements_per_group*2:elements_per_group*3],width = width, color = 'b')
ax[2,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1),eq_m3_15[elements_per_group*3:elements_per_group*4],yerr = eq_e3_15[elements_per_group*3:elements_per_group*4],width = width, color = '0')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[2,2].set_title("Signal to noise ratio = 10")






fig.legend(loc = 'center right')





#plt.savefig("eq_boosting_deter.pdf")
# %%
stn1 = pd.read_csv("boosting_nonlin100_4_0.3_3.csv")
stn2 = pd.read_csv("boosting_nonlin100_4_0.62_3.csv")
stn3 = pd.read_csv("boosting_nonlin100_4_0.9_3.csv")

stn1_10 = pd.read_csv("boosting_nonlin500_4_0.3_3.csv")
stn2_10 = pd.read_csv("boosting_nonlin500_4_0.62_3.csv")
stn3_10 = pd.read_csv("boosting_nonlin500_4_0.9_3.csv")

stn1_15 = pd.read_csv("boosting_nonlin1500_4_0.3_3.csv")
stn2_15 = pd.read_csv("boosting_nonlin1500_4_0.62_3.csv")
stn3_15 = pd.read_csv("boosting_nonlin1500_4_0.9_3.csv")


#%%
stn1 = pd.read_csv("boosting100_3_0.552.csv")
stn2 = pd.read_csv("boosting100_3_1.252.csv")
stn3 = pd.read_csv("boosting100_3_1.82.csv")

stn1_10 = pd.read_csv("boosting500_3_0.552.csv")
stn2_10 = pd.read_csv("boosting500_3_1.252.csv")
stn3_10 = pd.read_csv("boosting500_3_1.82.csv")

stn1_15 = pd.read_csv("boosting1500_3_0.552.csv")
stn2_15 = pd.read_csv("boosting1500_3_1.252.csv")
stn3_15 = pd.read_csv("boosting1500_3_1.82.csv")


#%%
stn1 = pd.read_csv("boosting_nonlin100_4_0.3.csv")
stn2 = pd.read_csv("boosting_nonlin100_4_0.62.csv")
stn3 = pd.read_csv("boosting_nonlin100_4_0.9.csv")

stn1_10 = pd.read_csv("boosting_nonlin500_4_0.3.csv")
stn2_10 = pd.read_csv("boosting_nonlin500_4_0.62.csv")
stn3_10 = pd.read_csv("boosting_nonlin500_4_0.9.csv")

stn1_15 = pd.read_csv("boosting_nonlin1000_4_0.3.csv")
stn2_15 = pd.read_csv("boosting_nonlin1000_4_0.62.csv")
stn3_15 = pd.read_csv("boosting_nonlin1000_4_0.9.csv")

#%%
stn1 = pd.read_csv("rndm100_3_0.55rf_deter.csv")
stn2 = pd.read_csv("rndm100_3_1.25rf_deter.csv")
stn3 = pd.read_csv("rndm100_3_1.8rf_deter.csv")

stn1_10 = pd.read_csv("rndm500_3_0.55rf_deter.csv")
stn2_10 = pd.read_csv("rndm500_3_1.25rf_deter.csv")
stn3_10 = pd.read_csv("rndm500_3_1.8rf_deter.csv")

stn1_15 = pd.read_csv("rndm1000_3_0.55rf_deter.csv")
stn2_15 = pd.read_csv("rndm1000_3_1.25rf_deter.csv")
stn3_15 = pd.read_csv("rndm1000_3_1.8rf_deter.csv")