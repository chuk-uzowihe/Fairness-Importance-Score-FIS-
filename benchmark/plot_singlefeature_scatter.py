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
            dp_fis.append(stn1.iloc[i]['dp'])
            eqop_fis.append(stn1.iloc[i]['eq'])
            dp_null.append(stn2.iloc[i]['dp'])
            eqop_null.append(stn2.iloc[i]['eq'])
        
            
    
    return dp_fis,dp_null,eqop_fis,eqop_null

#%%
stn1 = pd.read_csv("null1000_30.81.csv")
stn2 = pd.read_csv("null100_30.81.csv")
stn3 = pd.read_csv("null1000_30.71.csv")
stn4 = pd.read_csv("null100_30.71.csv")
stn5 = pd.read_csv("null1000_30.61.csv")
stn6 = pd.read_csv("null100_30.61.csv")
stn7 = pd.read_csv("null1000_30.51.csv")
stn8 = pd.read_csv("null100_30.51.csv")

# %%
#feature = 15
dp_fis,dp_null,eqop_fis,eqop_null = single_stn(stn1,stn2)
dp_fis2,dp_null2,eqop_fis2,eqop_null2 = single_stn(stn3,stn4)
dp_fis3,dp_null3,eqop_fis3,eqop_null3 = single_stn(stn5,stn6)
dp_fis4,dp_null4,eqop_fis4,eqop_null4 = single_stn(stn7,stn8)
# %%
fontsize = 15
fig, ax = plt.subplots(2,2,figsize=(14,9),sharex=True,sharey=True)
x_axis = np.arange(0.1,0.6,0.1)
ax[0,0].set_title("Non-protected to Protected ratio = 4", fontsize = fontsize)
ax[0,1].set_title("Non-protected to Protected ratio = 2.33", fontsize = fontsize)
ax[1,0].set_title("Non-protected to Protected ratio = 1.5", fontsize = fontsize)
ax[1,1].set_title("Non-protected to Protected ratio = 1", fontsize = fontsize)
ax[0,0].scatter(x_axis,dp_fis,color = 'g',label = "n = 1000")
ax[0,0].scatter(x_axis,dp_null,color = 'r',label = "n = 100")
ax[0,1].scatter(x_axis,dp_fis2,color = 'g')
ax[0,1].scatter(x_axis,dp_null2,color = 'r')
ax[1,0].scatter(x_axis,dp_fis3,color = 'g')
ax[1,0].scatter(x_axis,dp_null3,color = 'r')
ax[1,1].scatter(x_axis,dp_fis4,color = 'g')
ax[1,1].scatter(x_axis,dp_null4,color = 'r')
fig.legend(fontsize = fontsize)
fig.supxlabel("Left node to total ratio", fontsize = fontsize)
fig.supylabel("Null Fairness", fontsize = fontsize)
plt.savefig("dp_Null_fairness_wrong.pdf")
# %%
stn1 = pd.read_csv("boosting_nonlin1000_4_0.9individual.csv")
stn2 = pd.read_csv("boosting_nonlin100_4_0.9individual.csv")
stn1 = pd.read_csv("boosting1000_3_1.8individual1.csv")
stn2 = pd.read_csv("boosting100_3_1.8individual1.csv")