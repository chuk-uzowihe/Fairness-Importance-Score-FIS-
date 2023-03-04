import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt 


#Make plots for exploring data simulations

#read in data
fair_500 = pd.read_csv('/Users/camillelittle/Desktop/fair_FIS/results/dp_tree_500.csv', header = None)
fair_500 = fair_500.rename(columns = {0 : "Fair"})
fair_1000 = pd.read_csv('/Users/camillelittle/Desktop/fair_FIS/results/dp_tree_1000.csv', header = None)
fair_1000 = fair_1000.rename(columns = {0 : "Fair"})


acc_500 = pd.read_csv('/Users/camillelittle/Desktop/fair_FIS/results/acc_tree_500.csv', header = None)
acc_500 = acc_500.rename(columns = {0 : "Acc"})
acc_1000 = pd.read_csv('/Users/camillelittle/Desktop/fair_FIS/results/acc_tree_1000.csv', header = None)
acc_1000 = acc_1000.rename(columns = {0 : "Acc"})


x = [1,2,3,4,5,6,7,8,9,10,11,12]


plt.bar(x, list(acc_500.iloc[:,0]), 0.4, label = 'Accuracy')
plt.bar(x, list(fair_500.iloc[:,0]), 0.4, label = 'Fairness')

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature and Accuracy Scores, n = 500")
plt.legend()
plt.show()
#plt.savefig('/Users/camillelittle/Desktop/fair_FIS/Figures/boost_500_results.png')



plt.bar(x, list(acc_1000.iloc[:,0]), 0.4, label = 'Accuracy')
plt.bar(x, list(fair_1000.iloc[:,0]), 0.4, label = 'Fairness')

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature and Accuracy Scores, n = 1000")
plt.legend()
plt.show()

#plt.savefig('/Users/camillelittle/Desktop/fair_FIS/Figures/boost_1000_results.pdf')





 