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
            dp_values[j].append(stn['dp'].iloc[i*elements_per_group*4 + j])
            eq_values[j].append(stn['eqop'].iloc[i*elements_per_group*4 + j])
            dp_root_values[j].append(stn['dp_root'].iloc[i*elements_per_group*4 + j])
            eq_root_values[j].append(stn['eqop_root'].iloc[i*elements_per_group*4 + j])

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
        
    return  dp_mean, dp_err, occ_dp_mean, occ_dp_err,eqop_mean, eqop_err,occ_eqop_mean, occ_eqop_err
#%%
stn1 = pd.read_csv("rndmdp_eq100_1.csv")
stn2 = pd.read_csv("rndmdp_eq100_2.csv")
stn3 = pd.read_csv("rndmdp_eq100_3.csv")

stn1_10 = pd.read_csv("rndmdp_eq500_1.csv")
stn2_10 = pd.read_csv("rndmdp_eq500_2.csv")
stn3_10 = pd.read_csv("rndmdp_eq500_3.csv")

stn1_15 = pd.read_csv("rndmdp_eq1000_1.csv")
stn2_15 = pd.read_csv("rndmdp_eq1000_2.csv")
stn3_15 = pd.read_csv("rndmdp_eq1000_3.csv")

stn1_20 = pd.read_csv("rndmdp_eq2000_1.csv")
stn2_20 = pd.read_csv("rndmdp_eq2000_2.csv")
stn3_20 = pd.read_csv("rndmdp_eq2000_3.csv")
#%%
dp_m1, dp_e1,dp_root_m1,dp_root_e1, eq_m1,eq_e1, eq_root_m1,eq_root_e1 =single_stn(12,10,stn1)
dp_m2, dp_e2,dp_root_m2,dp_root_e2, eq_m2,eq_e2, eq_root_m2,eq_root_e2 = single_stn(12,10,stn2)
dp_m3, dp_e3,dp_root_m3,dp_root_e3, eq_m3,eq_e3, eq_root_m3,eq_root_e3 = single_stn(12,10,stn3)


dp_m1_10, dp_e1_10,dp_root_m1_10,dp_root_e1_10, eq_m1_10,eq_e1_10, eq_root_m1_10,eq_root_e1_10 =single_stn(12,10,stn1_10)
dp_m2_10, dp_e2_10,dp_root_m2_10,dp_root_e2_10, eq_m2_10,eq_e2_10, eq_root_m2_10,eq_root_e2_10 = single_stn(12,10,stn2_10)
dp_m3_10, dp_e3_10,dp_root_m3_10,dp_root_e3_10, eq_m3_10,eq_e3_10, eq_root_m3_10,eq_root_e3_10 = single_stn(12,10,stn3_10)


dp_m1_15, dp_e1_15,dp_root_m1_15,dp_root_e1_15, eq_m1_15,eq_e1_15, eq_root_m1_15,eq_root_e1_15 =single_stn(12,10,stn1_15)
dp_m2_15, dp_e2_15,dp_root_m2_15,dp_root_e2_15, eq_m2_15,eq_e2_15, eq_root_m2_15,eq_root_e2_15 = single_stn(12,10,stn2_15)
dp_m3_15, dp_e3_15,dp_root_m3_15,dp_root_e3_15, eq_m3_15,eq_e3_15, eq_root_m3_15,eq_root_e3_15 = single_stn(12,10,stn3_15)

dp_m1_20, dp_e1_20,dp_root_m1_20,dp_root_e1_20, eq_m1_20,eq_e1_20, eq_root_m1_20,eq_root_e1_20 =single_stn(12,10,stn1_20)
dp_m2_20, dp_e2_20,dp_root_m2_20,dp_root_e2_20, eq_m2_20,eq_e2_20, eq_root_m2_20,eq_root_e2_20 = single_stn(12,10,stn2_20)
dp_m3_20, dp_e3_20,dp_root_m3_20,dp_root_e3_20, eq_m3_20,eq_e3_20, eq_root_m3_20,eq_root_e3_20 = single_stn(12,10,stn3_20)


#%%
#dp_m4, dp_e4, eq_m4,eq_e4,occd_m4, occd_e4,occe_m4, occe_e4 = single_stn_result(elements_per_group,10,stn4) 
elements_per_group = 3
width = 0.3
fontzise = 20
fig, ax = plt.subplots(4,3,figsize=(22,21),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("Fairness Importance Score(FIS)]", fontsize = fontzise)
fig.supxlabel("Feature", fontsize = fontzise)
ax[0,0].bar(np.arange(1,elements_per_group+1)-width,eq_m1[0:elements_per_group],yerr = eq_e1[0:elements_per_group],width = width, color = 'r', label = "Group 1 (z & y)")
ax[0,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m1[elements_per_group:elements_per_group*2],yerr = eq_e1[elements_per_group:elements_per_group*2],width = width, color = 'g', label = "Group 2 (z)")
ax[0,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m1[elements_per_group*2:elements_per_group*3],yerr = eq_e1[elements_per_group*2:elements_per_group*3],width = width, color = 'y', label = "Group 3 (y)")
ax[0,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m1[elements_per_group*3:elements_per_group*4],yerr = eq_e1[elements_per_group*3:elements_per_group*4],width = width, color = 'c', label = "Group 4 (-)")

ax[0,0].bar(np.arange(1,elements_per_group+1)+width,eq_root_m1[0:elements_per_group],yerr = eq_root_e1[0:elements_per_group],width = width, color = 'r',hatch='//', label = "Group 1 (z & y) root")
ax[0,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m1[elements_per_group:elements_per_group*2],yerr = eq_root_e1[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//', label = "Group 2 (z) root")
ax[0,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m1[elements_per_group*2:elements_per_group*3],yerr = eq_root_e1[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//', label = "Group 3 (y) root")
ax[0,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m1[elements_per_group*3:elements_per_group*4],yerr = eq_root_e1[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//', label = "Group 4 (-) root")
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[0,0].set_title("Signal to noise ratio = 1", fontsize = fontzise)
ax[0,0].set_ylabel("n = 100", fontsize = fontzise)

ax[0,1].bar(np.arange(1,elements_per_group+1)-width,eq_m2[0:elements_per_group],yerr = eq_e2[0:elements_per_group],width = width, color = 'r')
ax[0,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m2[elements_per_group:elements_per_group*2],yerr = eq_e2[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[0,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m2[elements_per_group*2:elements_per_group*3],yerr = eq_e2[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[0,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m2[elements_per_group*3:elements_per_group*4],yerr = eq_e2[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[0,1].bar(np.arange(1,elements_per_group+1)+width,eq_root_m2[0:elements_per_group],yerr = eq_root_e2[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[0,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m2[elements_per_group:elements_per_group*2],yerr = eq_root_e2[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[0,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m2[elements_per_group*2:elements_per_group*3],yerr = eq_root_e2[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[0,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m2[elements_per_group*3:elements_per_group*4],yerr = eq_root_e2[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[0,1].set_title("Signal to noise ratio = 5", fontsize= fontzise)
#ax[0,1].legend()

ax[0,2].bar(np.arange(1,elements_per_group+1)-width,eq_m3[0:elements_per_group],yerr = eq_e3[0:elements_per_group],width = width, color = 'r')
ax[0,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m3[elements_per_group:elements_per_group*2],yerr = eq_e3[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[0,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m3[elements_per_group*2:elements_per_group*3],yerr = eq_e3[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[0,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m3[elements_per_group*3:elements_per_group*4],yerr = eq_e3[elements_per_group*3:elements_per_group*4],width = width, color = 'c')

ax[0,2].bar(np.arange(1,elements_per_group+1)+width,eq_root_m3[0:elements_per_group],yerr = eq_root_e3[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[0,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m3[elements_per_group:elements_per_group*2],yerr = eq_root_e3[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[0,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m3[elements_per_group*2:elements_per_group*3],yerr = eq_root_e3[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[0,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m3[elements_per_group*3:elements_per_group*4],yerr = eq_root_e3[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[0,2].set_title("Signal to noise ratio = 10", fontsize = fontzise)
#ax[0].set_title("sample size = 500")


ax[1,0].bar(np.arange(1,elements_per_group+1)-width,eq_m1_10[0:elements_per_group],yerr = eq_e1_10[0:elements_per_group],width = width, color = 'r')
ax[1,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m1_10[elements_per_group:elements_per_group*2],yerr = eq_e1_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m1_10[elements_per_group*2:elements_per_group*3],yerr = eq_e1_10[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[1,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m1_10[elements_per_group*3:elements_per_group*4],yerr = eq_e1_10[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[1,0].bar(np.arange(1,elements_per_group+1)+width,eq_root_m1_10[0:elements_per_group],yerr = eq_root_e1_10[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[1,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m1_10[elements_per_group:elements_per_group*2],yerr = eq_root_e1_10[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[1,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m1_10[elements_per_group*2:elements_per_group*3],yerr = eq_root_e1_10[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[1,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m1_10[elements_per_group*3:elements_per_group*4],yerr = eq_root_e1_10[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[1,0].set_title("Signal to noise ratio = 1", fontsize = fontzise)
ax[1,0].set_ylabel("n = 500", fontsize = fontzise)

ax[1,1].bar(np.arange(1,elements_per_group+1)-width,eq_m2_10[0:elements_per_group],yerr = eq_e2_10[0:elements_per_group],width = width, color = 'r')
ax[1,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m2_10[elements_per_group:elements_per_group*2],yerr = eq_e2_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m2_10[elements_per_group*2:elements_per_group*3],yerr = eq_e2_10[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[1,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m2_10[elements_per_group*3:elements_per_group*4],yerr = eq_e2_10[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")

ax[1,1].bar(np.arange(1,elements_per_group+1)+width,eq_root_m2_10[0:elements_per_group],yerr = eq_root_e2_10[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[1,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m2_10[elements_per_group:elements_per_group*2],yerr = eq_root_e2_10[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[1,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m2_10[elements_per_group*2:elements_per_group*3],yerr = eq_root_e2_10[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[1,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m2_10[elements_per_group*3:elements_per_group*4],yerr = eq_root_e2_10[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[1,1].set_title("Signal to noise ratio = 5" , fontsize = fontzise)
#ax[0,1].legend()

ax[1,2].bar(np.arange(1,elements_per_group+1)-width,eq_m3_10[0:elements_per_group],yerr = eq_e3_10[0:elements_per_group],width = width, color = 'r')
ax[1,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m3_10[elements_per_group:elements_per_group*2],yerr = eq_e3_10[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[1,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m3_10[elements_per_group*2:elements_per_group*3],yerr = eq_e3_10[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[1,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m3_10[elements_per_group*3:elements_per_group*4],yerr = eq_e3_10[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[1,2].bar(np.arange(1,elements_per_group+1)+width,eq_root_m3_15[0:elements_per_group],yerr = eq_root_e3_15[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[1,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m3_15[elements_per_group:elements_per_group*2],yerr = eq_root_e3_15[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[1,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m3_15[elements_per_group*2:elements_per_group*3],yerr = eq_root_e3_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[1,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m3_15[elements_per_group*3:elements_per_group*4],yerr = eq_root_e3_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[1,2].set_title("Signal to noise ratio = 10", fontsize = fontzise)

#ax[1].set_title("sample size = 500")



ax[2,0].bar(np.arange(1,elements_per_group+1)-width,eq_m1_15[0:elements_per_group],yerr = eq_e1_15[0:elements_per_group],width = width, color = 'r')
ax[2,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m1_15[elements_per_group:elements_per_group*2],yerr = eq_e1_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m1_15[elements_per_group*2:elements_per_group*3],yerr = eq_e1_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[2,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m1_15[elements_per_group*3:elements_per_group*4],yerr = eq_e1_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[2,0].bar(np.arange(1,elements_per_group+1)+width,eq_root_m1_15[0:elements_per_group],yerr = eq_root_e1_15[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[2,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m1_15[elements_per_group:elements_per_group*2],yerr = eq_root_e1_15[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[2,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m1_15[elements_per_group*2:elements_per_group*3],yerr = eq_root_e1_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[2,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m1_15[elements_per_group*3:elements_per_group*4],yerr = eq_root_e1_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')



ax[2,0].set_title("Signal to noise ratio = 1", fontsize = fontzise)
ax[2,0].set_ylabel("n = 1000", fontsize = fontzise)

ax[2,1].bar(np.arange(1,elements_per_group+1)-width,eq_m2_15[0:elements_per_group],yerr = eq_e2_15[0:elements_per_group],width = width, color = 'r')
ax[2,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m2_15[elements_per_group:elements_per_group*2],yerr = eq_e2_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m2_15[elements_per_group*2:elements_per_group*3],yerr = eq_e2_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[2,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m2_15[elements_per_group*3:elements_per_group*4],yerr = eq_e2_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[2,1].bar(np.arange(1,elements_per_group+1)+width,eq_root_m2_15[0:elements_per_group],yerr = eq_root_e2_15[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[2,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m2_15[elements_per_group:elements_per_group*2],yerr = eq_root_e2_15[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[2,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m2_15[elements_per_group*2:elements_per_group*3],yerr = eq_root_e2_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[2,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m2_15[elements_per_group*3:elements_per_group*4],yerr = eq_root_e2_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[2,1].set_title("Signal to noise ratio = 5", fontsize = fontzise)
#ax[0,1].legend()

ax[2,2].bar(np.arange(1,elements_per_group+1)-width,eq_m3_15[0:elements_per_group],yerr = eq_e3_15[0:elements_per_group],width = width, color = 'r')
ax[2,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m3_15[elements_per_group:elements_per_group*2],yerr = eq_e3_15[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[2,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m3_15[elements_per_group*2:elements_per_group*3],yerr = eq_e3_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[2,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m3_15[elements_per_group*3:elements_per_group*4],yerr = eq_e3_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[2,2].bar(np.arange(1,elements_per_group+1)+width,eq_root_m3_15[0:elements_per_group],yerr = eq_root_e3_15[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[2,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m3_15[elements_per_group:elements_per_group*2],yerr = eq_root_e3_15[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[2,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m3_15[elements_per_group*2:elements_per_group*3],yerr = eq_root_e3_15[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[2,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m3_15[elements_per_group*3:elements_per_group*4],yerr = eq_root_e3_15[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[2,2].set_title("Signal to noise ratio = 10", fontsize = fontzise)



ax[3,0].bar(np.arange(1,elements_per_group+1)-width,eq_m1_20[0:elements_per_group],yerr = eq_e1_20[0:elements_per_group],width = width, color = 'r', label = "Group 1 (z & y)")
ax[3,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m1_20[elements_per_group:elements_per_group*2],yerr = eq_e1_20[elements_per_group:elements_per_group*2],width = width, color = 'g', label = "Group 2 (z)")
ax[3,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m1_20[elements_per_group*2:elements_per_group*3],yerr = eq_e1_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y', label = "Group 3 (y)")
ax[3,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m1_20[elements_per_group*3:elements_per_group*4],yerr = eq_e1_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c', label = "Group 4 (-)")

ax[3,0].bar(np.arange(1,elements_per_group+1)+width,eq_root_m1_20[0:elements_per_group],yerr = eq_root_e1_20[0:elements_per_group],width = width, color = 'r',hatch='//', label = "Group 1 (z & y) root")
ax[3,0].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m1_20[elements_per_group:elements_per_group*2],yerr = eq_root_e1_20[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//', label = "Group 2 (z) root")
ax[3,0].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m1_20[elements_per_group*2:elements_per_group*3],yerr = eq_root_e1_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//', label = "Group 3 (y) root")
ax[3,0].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m1_20[elements_per_group*3:elements_per_group*4],yerr = eq_root_e1_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//', label = "Group 4 (-) root")
#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[3,0].set_title("Signal to noise ratio = 1", fontsize = fontzise)
ax[3,0].set_ylabel("n = 2000", fontsize = fontzise)
ax[3,1].bar(np.arange(1,elements_per_group+1)-width,eq_m2_20[0:elements_per_group],yerr = eq_e2_20[0:elements_per_group],width = width, color = 'r')
ax[3,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m2_20[elements_per_group:elements_per_group*2],yerr = eq_e2_20[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[3,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m2_20[elements_per_group*2:elements_per_group*3],yerr = eq_e2_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[3,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m2_20[elements_per_group*3:elements_per_group*4],yerr = eq_e2_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c')
#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[3,1].bar(np.arange(1,elements_per_group+1)+width,eq_root_m2_20[0:elements_per_group],yerr = eq_root_e2_20[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[3,1].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m2_20[elements_per_group:elements_per_group*2],yerr = eq_root_e2_20[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[3,1].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m2_20[elements_per_group*2:elements_per_group*3],yerr = eq_root_e2_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[3,1].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m2_20[elements_per_group*3:elements_per_group*4],yerr = eq_root_e2_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
ax[3,1].set_title("Signal to noise ratio = 5", fontsize= fontzise)
#ax[0,1].legend()

ax[3,2].bar(np.arange(1,elements_per_group+1)-width,eq_m3_20[0:elements_per_group],yerr = eq_e3_20[0:elements_per_group],width = width, color = 'r')
ax[3,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)-width,eq_m3_20[elements_per_group:elements_per_group*2],yerr = eq_e3_20[elements_per_group:elements_per_group*2],width = width, color = 'g')
ax[3,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)-width,eq_m3_20[elements_per_group*2:elements_per_group*3],yerr = eq_e3_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y')
ax[3,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)-width,eq_m3_20[elements_per_group*3:elements_per_group*4],yerr = eq_e3_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c')

ax[3,2].bar(np.arange(1,elements_per_group+1)+width,eq_root_m3_20[0:elements_per_group],yerr = eq_root_e3_20[0:elements_per_group],width = width, color = 'r',hatch='//')
ax[3,2].bar(np.arange(elements_per_group+1,elements_per_group*2+1)+width,eq_root_m3_20[elements_per_group:elements_per_group*2],yerr = eq_root_e3_20[elements_per_group:elements_per_group*2],width = width, color = 'g',hatch='//')
ax[3,2].bar(np.arange(elements_per_group*2+1,elements_per_group*3+1)+width,eq_root_m3_20[elements_per_group*2:elements_per_group*3],yerr = eq_root_e3_20[elements_per_group*2:elements_per_group*3],width = width, color = 'y',hatch='//')
ax[3,2].bar(np.arange(elements_per_group*3+1,elements_per_group*4+1)+width,eq_root_m3_20[elements_per_group*3:elements_per_group*4],yerr = eq_root_e3_20[elements_per_group*3:elements_per_group*4],width = width, color = 'c',hatch='//')
#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[3,2].set_title("Signal to noise ratio = 10", fontsize = fontzise)




fig.legend(loc = 'center right')

plt.savefig("eq_rndm_dt_1.pdf")

# %%
