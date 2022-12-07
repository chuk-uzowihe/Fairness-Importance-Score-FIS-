#%%
from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def single_stn(stn1,stn2):
    #stn1 = pd.read_csv("result_2_0.1.csv")
    #stn2 = pd.read_csv("result_2_0.7.csv")
    #stn3 = pd.read_csv("result_2_0.95.csv")
    #stn4 = pd.read_csv("result_2_1.2.csv")
    
    dp_fis = []
    eqop_fis = []
    dp_null = []
    eqop_null = []
    

    for i in range(len(stn1)):
            dp_fis.append(stn1.iloc[i]['protected'])
            eqop_fis.append(stn1.iloc[i]['else'])
            dp_null.append(stn2.iloc[i]['protected'])
            eqop_null.append(stn2.iloc[i]['else'])
    return dp_fis,dp_null,eqop_fis, eqop_null

#%%
stn1 = pd.read_csv("expected1000_3.csv")
stn2 = pd.read_csv("expected100_3.csv")


# %%
#feature = 15
dp_fis,dp_null,eq_fis,eq_null = single_stn(stn1,stn2)

# %%
fontsize = 15
fig, ax = plt.subplots(1,1,figsize=(14,9),sharex=True,sharey=True)
x_axis = np.arange(0.5,0.9,0.1)

ax.scatter(eq_fis,dp_fis,color = 'g',label = "n = 1000")
ax.scatter(eq_null,dp_null,color = 'r',label = "n = 100")
fig.legend(fontsize = fontsize)
fig.supxlabel("Left node to total ratio", fontsize = fontsize)
fig.supylabel("Null Fairness", fontsize = fontsize)
#plt.savefig("eq_Null_fairness1.pdf")
# %%
stn1 = pd.read_csv("boosting_nonlin1000_4_0.9individual.csv")
stn2 = pd.read_csv("boosting_nonlin100_4_0.9individual.csv")
stn1 = pd.read_csv("boosting1000_3_1.8individual1.csv")
stn2 = pd.read_csv("boosting100_3_1.8individual1.csv")