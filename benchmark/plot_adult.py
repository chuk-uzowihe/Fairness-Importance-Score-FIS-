indexes = np.nonzero(fis_dp)
dps = fis_dp[indexes]
imps = feature_imp[indexes]
names = column_names
# %%
sns.set_context('talk')
fontsize = 20
width = 0.5
x_axis = np.arange(1,len(dps)+1)
fig, ax = plt.subplots(1,1, figsize=(10, 8),sharex=True)
ax.bar(x_axis - width/2, dps,color = 'black', label = 'Fair FIS',width = width)
ax.bar(x_axis + width/2, imps,color = 'grey', label = 'FIS',width = width)
ax.set_xticklabels(names, fontsize= fontsize, rotation=90)
ax.set_xticks(list(range(1,len(dps)+1)))
ax.set_ylabel("Importance Score", fontsize = fontsize)
ax.set_ylabel("Feature", fontsize = fontsize)

fig.legend()
plt.savefig("output_orig.pdf")
plt.show()# %%