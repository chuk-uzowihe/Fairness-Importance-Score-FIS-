#%%

from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn(stn1,iterations, depth):
    dp_values = {}
    eq_values = {}
    [dp_values.setdefault(i, []) for i in range(depth)]
    [eq_values.setdefault(i, []) for i in range(depth)]
    for i in range(depth):
        for j in range(iterations):
            dp_values[j].append(stn1['dp'].iloc[i*j + j])
            eq_values[j].append(stn1['eqop'].iloc[i*j + j])
            
    dp_mean = []
    dp_err = []
    eqop_mean = []
    eqop_err = []
    
    for i in range(depth):
        dp_mean.append(np.mean(dp_values[i]))
        dp_err.append(np.var(dp_values[i]))
        eqop_mean.append(np.mean(eq_values[i]))
        eqop_err.append(np.var(eq_values[i]))
    return dp_mean, dp_err,eqop_mean, eqop_err
#%%
stn1 = pd.read_csv("depthdp_eq1000_1.csv")
stn2 = pd.read_csv("depthdp_eq1000_5.csv")
stn3 = pd.read_csv("depthdp_eq1000_10.csv")


stn1_10 = pd.read_csv("depthdp_eq500_1.csv")
stn2_10 = pd.read_csv("depthdp_eq500_5.csv")
stn3_10 = pd.read_csv("depthdp_eq500_10.csv")


stn1_15 = pd.read_csv("depthdp_eq1000_1.csv")
stn2_15 = pd.read_csv("depthdp_eq1000_5.csv")
stn3_15 = pd.read_csv("depthdp_eq1000_10.csv")
dp_m1, dp_e1, eq_m1,eq_e1 = single_stn(stn1 ,10, 10)
dp_m2, dp_e2, eq_m2,eq_e2 = single_stn(stn2 ,10, 10)
dp_m3, dp_e3, eq_m3,eq_e3 = single_stn(stn3 ,10, 10)


dp_m1_10, dp_e1_10, eq_m1_10,eq_e1_10 = single_stn(stn1_10,10, 10)
dp_m2_10, dp_e2_10, eq_m2_10,eq_e2_10 = single_stn(stn2_10,10, 10)
dp_m3_10, dp_e3_10, eq_m3_10,eq_e3_10 = single_stn(stn3_10,10, 10)


dp_m1_15, dp_e1_15, eq_m1_15,eq_e1_15 = single_stn(stn1_15,10, 10)
dp_m2_15, dp_e2_15, eq_m2_15,eq_e2_15 = single_stn(stn2_15,10, 10)
dp_m3_15, dp_e3_15, eq_m3_15,eq_e3_15 = single_stn(stn3_15,10, 10)


#%%

#dp_m4, dp_e4, eq_m4,eq_e4,occd_m4, occd_e4,occe_m4, occe_e4 = single_stn_result(elements_per_group,10,stn4) 
elements_per_group = 3
width = 0.3
fontsize = 20

fig, ax = plt.subplots(3,3,figsize=(22,18),sharex=True,sharey=True)
x_axis = np.arange(1,4*elements_per_group + 1)
fig.supylabel("fairness(DP)", fontsize = fontsize)
fig.supxlabel("Depth", fontsize = fontsize)

ax[0,0].bar(np.arange(1,7),eq_m1[0:6],yerr = eq_e1[0:6],width = width, color = 'r')

#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[0,0].set_title("Signal to noise ratio = 1", fontsize = fontsize)
ax[0,0].set_ylabel("n = 100", fontsize = 20)

ax[0,1].bar(np.arange(1,7),eq_m2[0:6],yerr = eq_e2[0:6],width = width, color = 'r')

#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[0,1].set_title("Signal to noise ratio = 5", fontsize = fontsize)
#ax[0,1].legend()

ax[0,2].bar(np.arange(1,7),eq_m3[0:6],yerr = eq_e3[0:6],width = width, color = 'r')

#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[0,2].set_title("Signal to noise ratio = 10", fontsize = fontsize)
#ax[0].set_title("sample size = 500")


ax[1,0].bar(np.arange(1,7),eq_m1_10[0:6],yerr = eq_e1_10[0:6],width = width, color = 'r')

#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[1,0].set_title("Signal to noise ratio = 1",fontsize = fontsize)
ax[1,0].set_ylabel("n = 500", fontsize = 20)

ax[1,1].bar(np.arange(1,7),eq_m2_10[0:6],yerr = eq_e2_10[0:6],width = width, color = 'r')

#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[1,1].set_title("Signal to noise ratio = 5", fontsize = fontsize)
#ax[0,1].legend()

ax[1,2].bar(np.arange(1,7),eq_m3_10[0:6],yerr = eq_e3_10[0:6],width = width, color = 'r')

#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[1,2].set_title("Signal to noise ratio = 10",fontsize = fontsize)

#ax[1].set_title("sample size = 500")



ax[2,0].bar(np.arange(1,7),eq_m1_15[0:6],yerr = eq_e1_15[0:6],width = width, color = 'r')

#ax[0,0].bar(x_axis,occd_m1,yerr = occd_e1,width = width, color = 'b', label = "OFS")
ax[2,0].set_title("Signal to noise ratio = 1")
ax[2,0].set_ylabel("n = 1000", fontsize = 20)

ax[2,1].bar(np.arange(1,7),eq_m2_15[0:6],yerr = eq_e2_15[0:6],width = width, color = 'r')

#ax[0,1].bar(x_axis,occd_m2,yerr = occd_e2,width = width, color = 'b', label = "OFS")
ax[2,1].set_title("Signal to noise ratio = 5",fontsize = fontsize)
#ax[0,1].legend()

ax[2,2].bar(np.arange(1,7),eq_m3_15[0:6],yerr = eq_e3_15[0:6],width = width, color = 'r')

#ax[1,0].bar(x_axis,occd_m3,yerr = occd_e3,width = width, color = 'b', label = "OFS")
ax[2,2].set_title("Signal to noise ratio = 10",fontsize = fontsize)


#fig.legend(loc = 'center right', fontsize = fontsize)

plt.savefig("depth_dp.pdf")

# %%
