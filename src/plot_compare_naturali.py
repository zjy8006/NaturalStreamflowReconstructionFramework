import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


xunhua = pd.read_csv('data/xunhua_seasonal_naturalization.csv',index_col='YQ')
guide = pd.read_csv('data/guide_seasonal_naturalization.csv',index_col='YQ')


guide_obs = guide['natural_flow']
xunhua_obs = xunhua['natural_flow']


guide_lstm = guide['VIF_LSTM']
xunhua_lstm = xunhua['VIF_LSTM']

guide_xgb = guide['VIF_XGB']
xunhua_xgb = xunhua['VIF_XGB']

guide_mlr = guide['VIF_MLR']
xunhua_mlr = xunhua['VIF_MLR']

guide_error_xgb = guide['natural_flow'] - guide['VIF_XGB']
xunhua_error_xgb = xunhua['natural_flow'] - xunhua['VIF_XGB']

guide_error_lstm = guide['natural_flow'] - guide['VIF_LSTM']
xunhua_error_lstm = xunhua['natural_flow'] - xunhua['VIF_LSTM']

guide_error_mlr = guide['natural_flow'] - guide['VIF_MLR']
xunhua_error_mlr = xunhua['natural_flow'] - xunhua['VIF_MLR']


guide_err_df = pd.DataFrame({
    'VIF-XGB':guide_error_xgb.values,
    'VIF-LSTM':guide_error_lstm.values,
    'VIF-MLR':guide_error_mlr.values
})


xunhua_err_df = pd.DataFrame({
    'VIF-XGB':xunhua_error_xgb.values,
    'VIF-LSTM':xunhua_error_lstm.values,
    'VIF-MLR':xunhua_error_mlr.values
})

ftsize = 8
fig = plt.figure(figsize=(7.48, 5.48))  # Adjusted figure size for 2x3 layout

# Create 2x3 subplot grid with equal aspect ratio
ax11 = plt.subplot2grid((2, 3), (0, 0), aspect='equal')
ax12 = plt.subplot2grid((2, 3), (0, 1), aspect='equal')
ax13 = plt.subplot2grid((2, 3), (0, 2), aspect='equal')
ax21 = plt.subplot2grid((2, 3), (1, 0), aspect='equal')
ax22 = plt.subplot2grid((2, 3), (1, 1), aspect='equal')
ax23 = plt.subplot2grid((2, 3), (1, 2), aspect='equal')

# Create mask for pre/post 2000 split
guide_pre2000 = guide_obs.index <= '2000-4'
guide_post2000 = guide_obs.index > '2000-4'
xunhua_pre2000 = xunhua_obs.index <= '2000-4'
xunhua_post2000 = xunhua_obs.index > '2000-4'

# Guide station XGB comparison (top left)
sns.scatterplot(x=guide_xgb[guide_pre2000], y=guide_obs[guide_pre2000], 
                label='1986-2000', ax=ax11, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=guide_xgb[guide_post2000], y=guide_obs[guide_post2000], 
                label='2001-2018', ax=ax11, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax11.get_xlim(), ax11.get_ylim()]),
    np.max([ax11.get_xlim(), ax11.get_ylim()]),
]
ax11.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Guide station LSTM comparison (top middle)
sns.scatterplot(x=guide_lstm[guide_pre2000], y=guide_obs[guide_pre2000], 
                label='1986-2000', ax=ax12, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=guide_lstm[guide_post2000], y=guide_obs[guide_post2000], 
                label='2001-2018', ax=ax12, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax12.get_xlim(), ax12.get_ylim()]),
    np.max([ax12.get_xlim(), ax12.get_ylim()]),
]
ax12.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Guide station MLR comparison (top right)
sns.scatterplot(x=guide_mlr[guide_pre2000], y=guide_obs[guide_pre2000], 
                label='1986-2000', ax=ax13, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=guide_mlr[guide_post2000], y=guide_obs[guide_post2000], 
                label='2001-2018', ax=ax13, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax13.get_xlim(), ax13.get_ylim()]),
    np.max([ax13.get_xlim(), ax13.get_ylim()]),
]
ax13.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Xunhua station XGB comparison (bottom left)
sns.scatterplot(x=xunhua_xgb[xunhua_pre2000], y=xunhua_obs[xunhua_pre2000], 
                ax=ax21, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=xunhua_xgb[xunhua_post2000], y=xunhua_obs[xunhua_post2000], 
                ax=ax21, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax21.get_xlim(), ax21.get_ylim()]),
    np.max([ax21.get_xlim(), ax21.get_ylim()]),
]
ax21.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Xunhua station LSTM comparison (bottom middle)
sns.scatterplot(x=xunhua_lstm[xunhua_pre2000], y=xunhua_obs[xunhua_pre2000], 
                ax=ax22, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=xunhua_lstm[xunhua_post2000], y=xunhua_obs[xunhua_post2000], 
                ax=ax22, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax22.get_xlim(), ax22.get_ylim()]),
    np.max([ax22.get_xlim(), ax22.get_ylim()]),
]
ax22.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Xunhua station MLR comparison (bottom right)
sns.scatterplot(x=xunhua_mlr[xunhua_pre2000], y=xunhua_obs[xunhua_pre2000], 
                ax=ax23, color='blue', marker='o', s=20, zorder=2)
sns.scatterplot(x=xunhua_mlr[xunhua_post2000], y=xunhua_obs[xunhua_post2000], 
                ax=ax23, color='red', marker='o', s=20, zorder=2)
lims = [
    np.min([ax23.get_xlim(), ax23.get_ylim()]),
    np.max([ax23.get_xlim(), ax23.get_ylim()]),
]
ax23.plot(lims, lims, 'k--', alpha=0.75, zorder=1)

# Update labels
ax11.set_xlabel('VIF-XGB($m^3/s$)', fontsize=ftsize)
ax12.set_xlabel('VIF-LSTM($m^3/s$)', fontsize=ftsize)
ax13.set_xlabel('VIF-MLR($m^3/s$)', fontsize=ftsize)
ax21.set_xlabel('VIF-XGB($m^3/s$)', fontsize=ftsize)
ax22.set_xlabel('VIF-LSTM($m^3/s$)', fontsize=ftsize)
ax23.set_xlabel('VIF-MLR($m^3/s$)', fontsize=ftsize)

# Set y-labels for all subplots
ax11.set_ylabel('Water Balance($m^3/s$)', fontsize=ftsize)
ax12.set_ylabel('', fontsize=ftsize)
ax13.set_ylabel('', fontsize=ftsize)
ax21.set_ylabel('Water Balance($m^3/s$)', fontsize=ftsize)
ax22.set_ylabel('', fontsize=ftsize)
ax23.set_ylabel('', fontsize=ftsize)

# Remove grid and vertical lines since they're not needed for scatter comparison
for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
    ax.grid(False)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='both', useMathText=True)

# Move subplot labels to upper left corner
for ax, fig_id in zip([ax11, ax12, ax13, ax21, ax22, ax23], ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.05 * (xlim[1] - xlim[0])  # Adjust x position
    y_pos = ylim[1] - 0.05 * (ylim[1] - ylim[0])  # Adjust y position
    ax.text(x_pos, y_pos, fig_id, ha='left', va='top', fontsize=ftsize+2, fontweight='bold')

# Handle legend
handles, labels = [], []
for ax in [ax11, ax12, ax13, ax21, ax22, ax23]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:  # Avoid duplicate legend entries
            handles.append(handle)
            labels.append(label)
    legend = ax.legend()
    legend.remove()

fig.legend(handles, labels, 
          loc='upper center', 
          ncol=3, 
          fontsize=ftsize, 
          frameon=False)

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.94, 
                   hspace=0.3, wspace=0.25)  # Increased spacing between subplots

# Save figures with new dimensions
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/compare_naturalization_results.tif', 
            format='TIFF', dpi=500, transparent=True, bbox_inches='tight')

# Create new figure for boxplot
fig2 = plt.figure(figsize=(7.48, 4))

# Calculate deviations
guide_deviations = pd.DataFrame({
    'VIF-XGB': guide_obs - guide_xgb,
    'VIF-LSTM': guide_obs - guide_lstm,
    'VIF-MLR': guide_obs - guide_mlr
})

xunhua_deviations = pd.DataFrame({
    'VIF-XGB': xunhua_obs - xunhua_xgb,
    'VIF-LSTM': xunhua_obs - xunhua_lstm,
    'VIF-MLR': xunhua_obs - xunhua_mlr
})

# Create subplot grid
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# Create boxplots with individual points
sns.boxplot(data=guide_deviations, ax=ax1, width=0.5, color='lightgray')
sns.stripplot(data=guide_deviations, ax=ax1, size=3, color='black', alpha=0.3, jitter=0.2)

sns.boxplot(data=xunhua_deviations, ax=ax2, width=0.5, color='lightgray')
sns.stripplot(data=xunhua_deviations, ax=ax2, size=3, color='black', alpha=0.3, jitter=0.2)

# Customize plots
ax1.set_title('Guide Station', fontsize=ftsize)
ax2.set_title('Xunhua Station', fontsize=ftsize)

ax1.set_ylabel('Deviation($m^3/s$)', fontsize=ftsize)
ax2.set_ylabel('')

ax1.set_xlabel('Model', fontsize=ftsize)
ax2.set_xlabel('Model', fontsize=ftsize)

# Add subplot labels
ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, 
         fontsize=ftsize+2, fontweight='bold', va='top')
ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, 
         fontsize=ftsize+2, fontweight='bold', va='top')

# Format scientific notation
for ax in [ax1, ax2]:
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y', useMathText=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.15)

# Save figures
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/model_deviations_boxplot.tif', 
            format='TIFF', dpi=500, transparent=True, bbox_inches='tight')
plt.show()
