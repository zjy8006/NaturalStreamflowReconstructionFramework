import matplotlib.pyplot as plt
import pandas as pd
import calendar

guide_lstm = pd.read_csv('data/naturalization_1957_2018_viflstm_guide.csv',parse_dates=['date'],index_col='date')
guide_mlr = pd.read_csv('data/naturalization_1957_2018_vifmlr_guide.csv',parse_dates=['date'],index_col='date')

guide_lstm = guide_lstm.loc['1958-01-01':'2018-12-31',['flow','VIF_LSTM']]
guide_mlr = guide_mlr.loc['1958-01-01':'2018-12-31',['VIF_MLR']]
guide = pd.concat([guide_lstm,guide_mlr],axis=1)

xunhua_lstm = pd.read_csv('data/naturalization_1957_2018_viflstm_xunhua.csv',parse_dates=['date'],index_col='date')
xunhua_mlr = pd.read_csv('data/naturalization_1957_2018_vifmlr_xunhua.csv',parse_dates=['date'],index_col='date')

xunhua_lstm = xunhua_lstm.loc['1958-01-01':'2018-12-31',['flow','VIF_LSTM']]
xunhua_mlr = xunhua_mlr.loc['1958-01-01':'2018-12-31',['VIF_MLR']]
xunhua = pd.concat([xunhua_lstm,xunhua_mlr],axis=1)

print(guide['flow']-xunhua['flow'])

ftsize=10
fig = plt.figure(figsize=(7.48,5.5))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(guide.index,guide['flow'],color='gray',label='Observed flow at Guide\nCal period: 1958~1981\nTest period: 1982~1985\nNat period:1986~2018',zorder=1)
ax1.plot(guide.index,guide['VIF_LSTM'],color='blue',label='VIF-LSTM\nCal NNSE=0.99\ntest NNSE=0.99',zorder=0)
ax1.plot(guide.index,guide['VIF_MLR'],color='red',label='VIF-MLR\nCal NNSE=0.99\ntest NNSE=0.99',zorder=0)

ax2.plot(xunhua.index,xunhua['flow'],color='gray',label='Observed flow at Xunhua\nCal period: 1958~1981\nTest period: 1982~1985\nNat period:1986~2018',zorder=1)
ax2.plot(xunhua.index,xunhua['VIF_LSTM'],color='blue',label='VIF-LSTM\nCal NNSE=0.98\ntest NNSE=0.99',zorder=0)
ax2.plot(xunhua.index,xunhua['VIF_MLR'],color='red',label='VIF-MLR\nCal NNSE=0.99\ntest NNSE=0.98',zorder=0)

ax2.set_xlabel('Date(month)',fontsize=ftsize)
ax1.set_ylabel('Runoff($10^8m^3$)',fontsize=ftsize)
ax2.set_ylabel('Runoff($10^8m^3$)',fontsize=ftsize)

# ax1.axvline(x=pd.to_datetime('1983-01-31'),ymin=50,ymax=3500,color='black',linestyle='--')
# ax1.axvline(x=pd.to_datetime('1986-01-31'),ymin=50,ymax=3500,color='black',linestyle='--')

# ax1.vlines(x=pd.to_datetime('1983-01-31'),ymin=50,ymax=3800,color='black',linestyle='--')
# ax1.vlines(x=pd.to_datetime('1986-01-31'),ymin=50,ymax=3800,color='black',linestyle='--')
# ax1.hlines(y=3500,xmin=pd.to_datetime('1958-01-01'),xmax=pd.to_datetime('1983-12-31'),color='black',linestyle='--')

ax1.set_ylim([50,5000])
ax2.set_ylim([50,5000])

ax1.legend(loc='upper left', ncol=3,shadow=False,frameon=False,fontsize=ftsize)
ax2.legend(loc='upper left', ncol=3,shadow=False,frameon=False,fontsize=ftsize)
plt.subplots_adjust(left=0.10, bottom=0.09, right=0.99,top=0.98, hspace=0.15, wspace=0.2)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/naturalization_1958_2018.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()