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

guide_mlr = guide['VIF_MLR']
xunhua_mlr = xunhua['VIF_MLR']

guide_error_lstm = guide['natural_flow'] - guide['VIF_LSTM']
xunhua_error_lstm = xunhua['natural_flow'] - xunhua['VIF_LSTM']

guide_error_mlr = guide['natural_flow'] - guide['VIF_MLR']
xunhua_error_mlr = xunhua['natural_flow'] - xunhua['VIF_MLR']

guide_err_df = pd.DataFrame({
    'VIF-LSTM':guide_error_lstm.values,
    'VIF-MLR':guide_error_mlr.values
})


xunhua_err_df = pd.DataFrame({
    'VIF-LSTM':xunhua_error_lstm.values,
    'VIF-MLR':xunhua_error_mlr.values
})

ftsize=10
fig = plt.figure(figsize=(7.48,5))

# add subplots with the first axes span two column and the seconed span one column
ax11  = plt.subplot2grid((2,3), (0,0), colspan=2,)#ssa
ax12  = plt.subplot2grid((2,3), (0,2), colspan=1,)
ax21  = plt.subplot2grid((2,3), (1,0), colspan=2,)
ax22  = plt.subplot2grid((2,3), (1,2), colspan=1,)


# ax21 = fig.add_subplot(2,2,3)
# ax22 = fig.add_subplot(2,2,4)


sns.lineplot(x=guide_obs.index, y=guide_obs.values,label='Natural flow',ax=ax11,color='blue',zorder=1)
sns.lineplot(x=guide_lstm.index, y=guide_lstm.values,label='VIF-LSTM',ax=ax11,color='red',zorder=3)
sns.lineplot(x=guide_mlr.index, y=guide_mlr.values,label='VIF-MLR',ax=ax11,color='cyan',zorder=2)
# sns.scatterplot(x=guide_lstm.index, y=guide_lstm.values,s=15,marker='o',label='VIF-LSTM',ax=ax11,color='tab:red',zorder=3)
# sns.scatterplot(x=guide_mlr.index, y=guide_mlr.values,s=15,marker='s',label='VIF-MLR',ax=ax11,color='tab:cyan',zorder=2)

# sns.boxplot(guide_err_df,ax=ax12)
# sns.boxplot(xunhua_err_df,ax=ax22)

sns.violinplot(guide_err_df,ax=ax12)
sns.violinplot(xunhua_err_df,ax=ax22)


sns.lineplot(x=xunhua_obs.index, y=xunhua_obs.values,ax=ax21,color='blue',zorder=1)
sns.lineplot(x=xunhua_lstm.index, y=xunhua_lstm.values,ax=ax21,color='red',zorder=3)
sns.lineplot(x=xunhua_mlr.index, y=xunhua_mlr.values,ax=ax21,color='tab:cyan',zorder=2)
# sns.scatterplot(x=xunhua_lstm.index, y=xunhua_lstm.values,s=15,marker='o',ax=ax21,color='tab:red',zorder=3)
# sns.scatterplot(x=xunhua_mlr.index, y=xunhua_mlr.values,s=15,marker='s',ax=ax21,color='tab:cyan',zorder=2)



tick_positions = np.arange(0, len(guide_obs.index), 6)  # 每5个索引一个标签
tick_labels = [guide_obs.index[i] for i in tick_positions]  # 从标签列表中选取这些标签
ax11.set_ylabel('Flow($m^3/s$)', fontsize=ftsize)

ax11.set_xticks([])

ax12.set_ylabel('Deviation($m^3/s$)', fontsize=ftsize)
ax22.set_ylabel('Deviation($m^3/s$)', fontsize=ftsize)

ax11.get_xaxis().set_visible(False)
ax12.get_xaxis().set_visible(False)
ax21.set_ylabel('Flow($m^3/s$)', fontsize=ftsize)
ax21.set_xticks(ticks=tick_positions, labels=tick_labels, rotation=45, fontsize=ftsize)
ax21.set_xlabel('Date(Year-Season)', fontsize=ftsize)
ax22.set_xlabel('Model', fontsize=ftsize)
ax22.set_xticks(ticks=[0,1],labels=['VIF-LSTM','VIF-MLR'],rotation=33, fontsize=ftsize)

ax11.axvline(x='2000-4',ls='--',color='black',zorder=0)
ax21.axvline(x='2000-4',ls='--',color='black',zorder=0)

for ax in [ax11,ax21]:
    ylim = ax.get_ylim()
    y_pos = ylim[1] - 0.1 * (ylim[1] - ylim[0])
    ax.text('2001-1', y_pos, 'Winter of 2000', ha='left', va='bottom')

for ax in [ax11,ax12,ax21,ax22]:
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)



for ax, fig_id in zip([ax11,ax12,ax21,ax22],['(a)','(b)','(c)','(d)']):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.02 * (xlim[1] - xlim[0])
    y_pos = ylim[0] + 0.02 * (ylim[1] - ylim[0])
    ax.text(x_pos, y_pos, fig_id, ha='left', va='bottom')


for ax in [ax11,ax12,ax21,ax22]:
    legend = ax.legend()
    legend.remove()
# 获取所有图例
handles, labels = [], []

# 获取主坐标轴的图例
for ax in [ax11,ax12,ax21,ax22]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)


fig.legend(handles, labels, 
           loc='upper center', 
        #    bbox_to_anchor=(0.5, 1.0), 
           ncol=3, 
           fontsize=ftsize, 
           frameon=False)


plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99,top=0.90, hspace=0.15, wspace=0.30)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/compare_naturalization_results.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')
plt.show()
