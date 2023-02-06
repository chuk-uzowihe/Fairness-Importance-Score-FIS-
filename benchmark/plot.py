#%%
from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
boosting = pd.read_csv("result_boosting_german1.csv")
rf = pd.read_csv("result_rf_german1.csv")
dt = pd.read_csv("result_tree_german1.csv")

#%%
width = 0.33
boost = boosting['fis_dp']
rf = rf['fis_dp']
dt = dt['fis_dp']
x = np.arange(len(boost))
plt.bar(x-width,boost,color = 'r',width = width)
plt.bar(x,rf,color = 'g',width = width)
plt.bar(x+width,dt,color = 'b',width = width)
plt.show()

# %%
