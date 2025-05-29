import matplotlib.pyplot as plt
import pandas as pd
import calendar

guide_lstm = pd.read_csv('data/naturalization_1957_2018_viflstm_guide.csv',parse_dates=['date'],index_col='date')
guide_mlr = pd.read_csv('data/naturalization_1957_2018_vifmlr_guide.csv',parse_dates=['date'],index_col='date')
guide_xgb = pd.read_csv('results/guide_vif_xgb_pred.csv',parse_dates=['date'],index_col='date')

guide_lstm = guide_lstm.loc['1958-01-01':'2018-12-31',['flow','VIF_LSTM']]
guide_mlr = guide_mlr.loc['1958-01-01':'2018-12-31',['VIF_MLR']]
guide_xgb = guide_xgb.loc['1958-01-01':'2018-12-31',['VIF-XGB']]
guide = pd.concat([guide_lstm,guide_mlr,guide_xgb],axis=1)

xunhua_lstm = pd.read_csv('data/naturalization_1957_2018_viflstm_xunhua.csv',parse_dates=['date'],index_col='date')
xunhua_mlr = pd.read_csv('data/naturalization_1957_2018_vifmlr_xunhua.csv',parse_dates=['date'],index_col='date')
xunhua_xgb = pd.read_csv('results/xunhua_vif_xgb_pred.csv',parse_dates=['date'],index_col='date')

xunhua_lstm = xunhua_lstm.loc['1958-01-01':'2018-12-31',['flow','VIF_LSTM']]
xunhua_mlr = xunhua_mlr.loc['1958-01-01':'2018-12-31',['VIF_MLR']]
xunhua_xgb = xunhua_xgb.loc['1958-01-01':'2018-12-31',['VIF-XGB']]
xunhua = pd.concat([xunhua_lstm,xunhua_mlr,xunhua_xgb],axis=1)

print(guide['flow']-xunhua['flow'])

ftsize = 8
# Modified figure size for 6x1 layout
fig = plt.figure(figsize=(7.48, 8.00))
ax1 = fig.add_subplot(6, 1, 1)
ax2 = fig.add_subplot(6, 1, 2)
ax3 = fig.add_subplot(6, 1, 3)
ax4 = fig.add_subplot(6, 1, 4)
ax5 = fig.add_subplot(6, 1, 5)
ax6 = fig.add_subplot(6, 1, 6)

marker_size = 3


# Guide station - VIF-MLR
ax1.plot(guide.index, guide['flow'], color='gray', label='Observed flow at Guide\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax1.scatter(guide.index, guide['VIF_MLR'], color='red', marker='^', s=marker_size, label='VIF-MLR\nCal NNSE=0.99; test NNSE=0.99', zorder=1)
ax1.text(0.02, 0.92, '(b)', transform=ax2.transAxes, fontsize=ftsize)

# Guide station - VIF-LSTM
ax2.plot(guide.index, guide['flow'], color='gray', label='Observed flow at Guide\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax2.scatter(guide.index, guide['VIF_LSTM'], color='blue', marker='o', s=marker_size, label='VIF-LSTM\nCal NNSE=0.99; test NNSE=0.99', zorder=1)
ax2.text(0.02, 0.92, '(a)', transform=ax1.transAxes, fontsize=ftsize)

# Guide station - VIF-XGB
ax3.plot(guide.index, guide['flow'], color='gray', label='Observed flow at Guide\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax3.scatter(guide.index, guide['VIF-XGB'], color='green', marker='s', s=marker_size, label='VIF-XGB\nCal NNSE=0.99; test NNSE=0.99', zorder=1)
ax3.text(0.02, 0.92, '(c)', transform=ax3.transAxes, fontsize=ftsize)

# Xunhua station - VIF-MLR
ax4.plot(xunhua.index, xunhua['flow'], color='gray', label='Observed flow at Xunhua\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax4.scatter(xunhua.index, xunhua['VIF_MLR'], color='red', marker='^', s=marker_size, label='VIF-MLR\nCal NNSE=0.99; test NNSE=0.98', zorder=1)
ax4.text(0.02, 0.92, '(e)', transform=ax5.transAxes, fontsize=ftsize)

# Xunhua station - VIF-LSTM
ax5.plot(xunhua.index, xunhua['flow'], color='gray', label='Observed flow at Xunhua\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax5.scatter(xunhua.index, xunhua['VIF_LSTM'], color='blue', marker='o', s=marker_size, label='VIF-LSTM\nCal NNSE=0.98; test NNSE=0.99', zorder=1)
ax5.text(0.02, 0.92, '(d)', transform=ax4.transAxes, fontsize=ftsize)



# Xunhua station - VIF-XGB
ax6.plot(xunhua.index, xunhua['flow'], color='gray', label='Observed flow at Xunhua\nCal: 1958~1981; Test: 1982~1985;Nat:1986~2018', zorder=0)
ax6.scatter(xunhua.index, xunhua['VIF-XGB'], color='green', marker='s', s=marker_size, label='VIF-XGB\nCal NNSE=0.99; test NNSE=1.00', zorder=1)
ax6.text(0.02, 0.92, '(f)', transform=ax6.transAxes, fontsize=ftsize)

# Hide x-axis ticks for first five subplots
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
ax4.set_xticklabels([])
ax5.set_xticklabels([])

# Set y-labels and x-label
ax1.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax2.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax3.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax4.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax5.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax6.set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
ax6.set_xlabel('Date(month)', fontsize=ftsize)

# Set y-limits for all subplots
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_ylim([50, 5000])
    ax.legend(loc='upper left', ncol=2, shadow=False, frameon=False, fontsize=ftsize)

# Adjusted subplot spacing for 6x1 layout
plt.subplots_adjust(left=0.10, bottom=0.05, right=0.95, top=0.98, hspace=0.05)

plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/naturalization_1958_2018.tif', 
            format='TIFF', dpi=500, transparent=True, bbox_inches='tight')

plt.show()