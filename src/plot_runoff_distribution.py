import pandas as pd
import calendar
import matplotlib.pyplot as plt

def Get_days_from_month(date):
    year = date.year
    month = date.month
    # year, month, day = map(int, date.split('-'))

    # 获取该月的天数
    days_in_month = calendar.monthrange(year, month)[1]

    return days_in_month

# 贵德影响期天然径流
guide_infl = pd.read_csv('data/guide_naturalized_monthly_flow.csv', parse_dates=['date'],index_col='date')
guide_infl['date_'] = guide_infl.index
guide_infl['days'] = guide_infl['date_'].apply(Get_days_from_month)
guide_infl['runoff(10^8m^3)'] = guide_infl['VIF-LSTM']*guide_infl['days']*24*3600/100000000
guide_infl['month'] = guide_infl['date_'].dt.month
guide_infl = guide_infl.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
guide_infl.set_index('month',inplace=True)
guide_infl['Ratio'] = guide_infl/guide_infl.sum()*100

# 贵德影响期实测径流
guide_infl_obs = pd.read_csv('data/guide_monthly_flow.csv', parse_dates=['date'],index_col='date')
guide_infl_obs = guide_infl_obs.loc['1986-01-01':'2018-12-01']
guide_infl_obs['date_'] = guide_infl_obs.index
guide_infl_obs['days'] = guide_infl_obs['date_'].apply(Get_days_from_month)
guide_infl_obs['runoff(10^8m^3)'] = guide_infl_obs['flow(m^3/s)']*guide_infl_obs['days']*24*3600/100000000
guide_infl_obs['month'] = guide_infl_obs['date_'].dt.month
guide_infl_obs = guide_infl_obs.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
guide_infl_obs.set_index('month',inplace=True)
guide_infl_obs['Ratio'] = guide_infl_obs/guide_infl_obs.sum()*100

# 贵德影响前期径流
guide_preinfl = pd.read_csv('data/guide_monthly_flow.csv', parse_dates=['date'],index_col='date')
# select rows where date column range from 1957-01-01 to 1985-12-01
guide_preinfl = guide_preinfl.loc['1957-01-01':'1985-12-01']
guide_preinfl['date_'] = guide_preinfl.index
guide_preinfl['days'] = guide_preinfl['date_'].apply(Get_days_from_month)
guide_preinfl['runoff(10^8m^3)'] = guide_preinfl['flow(m^3/s)']*guide_preinfl['days']*24*3600/100000000
guide_preinfl['month'] = guide_preinfl['date_'].dt.month
guide_preinfl = guide_preinfl.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
guide_preinfl.set_index('month',inplace=True)
guide_preinfl['Ratio'] = guide_preinfl/guide_preinfl.sum()*100


# 循化影响期天然径流
xunhua_infl = pd.read_csv('data/xunhua_naturalized_monthly_flow.csv', parse_dates=['date'],index_col='date')
xunhua_infl['date_'] = xunhua_infl.index
xunhua_infl['days'] = xunhua_infl['date_'].apply(Get_days_from_month)
xunhua_infl['runoff(10^8m^3)'] = xunhua_infl['VIF-LSTM']*xunhua_infl['days']*24*3600/100000000
xunhua_infl['month'] = xunhua_infl['date_'].dt.month
xunhua_infl = xunhua_infl.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
xunhua_infl.set_index('month',inplace=True)
xunhua_infl['Ratio'] = xunhua_infl/xunhua_infl.sum()*100

# 循化影响期实测径流
xunhua_infl_obs = pd.read_csv('data/xunhua_monthly_flow.csv', parse_dates=['date'],index_col='date')
xunhua_infl_obs = xunhua_infl_obs.loc['1986-01-01':'2018-12-01']
xunhua_infl_obs['date_'] = xunhua_infl_obs.index
xunhua_infl_obs['days'] = xunhua_infl_obs['date_'].apply(Get_days_from_month)
xunhua_infl_obs['runoff(10^8m^3)'] = xunhua_infl_obs['flow(m^3/s)']*xunhua_infl_obs['days']*24*3600/100000000
xunhua_infl_obs['month'] = xunhua_infl_obs['date_'].dt.month
xunhua_infl_obs = xunhua_infl_obs.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
xunhua_infl_obs.set_index('month',inplace=True)
xunhua_infl_obs['Ratio'] = xunhua_infl_obs/xunhua_infl_obs.sum()*100

# 循化影响前期径流
xunhua_preinfl = pd.read_csv('data/xunhua_monthly_flow.csv', parse_dates=['date'],index_col='date')
# select rows where date column range from 1957-01-01 to 1985-12-01
xunhua_preinfl = xunhua_preinfl.loc['1957-01-01':'1985-12-01']
xunhua_preinfl['date_'] = xunhua_preinfl.index
xunhua_preinfl['days'] = xunhua_preinfl['date_'].apply(Get_days_from_month)
xunhua_preinfl['runoff(10^8m^3)'] = xunhua_preinfl['flow(m^3/s)']*xunhua_preinfl['days']*24*3600/100000000
xunhua_preinfl['month'] = xunhua_preinfl['date_'].dt.month
xunhua_preinfl = xunhua_preinfl.groupby('month')['runoff(10^8m^3)'].mean().reset_index()
xunhua_preinfl.set_index('month',inplace=True)
xunhua_preinfl['Ratio'] = xunhua_preinfl/xunhua_preinfl.sum()*100


ftsize=10
natural_runoff_color = '#4CC9F0' #'skyblue'
ratio_natural_runoff_color = 'tomato'

naturalized_runoff_color = '#4361EE'
rario_naturalized_runoff_color = 'red'
observed_runoff_color = 'slategray'
ratio_observed_runoff_color = 'fuchsia'


# 给出结果
print(guide_infl)

# 创建一个绘图区域，包含两个子图
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7.48, 7.48))

# 子图1：标准期
axes[0][0].bar(guide_preinfl.index, guide_preinfl['runoff(10^8m^3)'], color=natural_runoff_color, label='Natural runoff')
axes[0][0].set_xlabel('Month', fontsize=ftsize)
axes[0][0].set_ylabel('runoff($10^8m^3$)', fontsize=ftsize)
axes[0][0].set_title('(a) Pre-influenced Period at Guide', fontsize=ftsize)

# 创建第二个y轴
ax2_1 = axes[0][0].twinx()
ax2_1.plot(guide_preinfl.index, guide_preinfl['Ratio'], color=ratio_natural_runoff_color, label='Ratio of annual natural runoff')
ax2_1.set_ylabel('Ratio (%)', fontsize=ftsize)

# 子图2：影响期
axes[0][1].bar(guide_infl.index - 0.2, guide_infl['runoff(10^8m^3)'], width=0.4, color=naturalized_runoff_color, label='Naturalized runoff')
axes[0][1].bar(guide_infl_obs.index + 0.2, guide_infl_obs['runoff(10^8m^3)'], width=0.4, color=observed_runoff_color, label='Observed runoff')
axes[0][1].set_xlabel('Month', fontsize=ftsize)
axes[0][1].set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
axes[0][1].set_title('(b) Influenced Period at Guide', fontsize=ftsize)

# 创建第二个y轴
ax2_2 = axes[0][1].twinx()
ax2_2.plot(guide_infl.index, guide_infl['Ratio'], color=rario_naturalized_runoff_color, label='Ratio of annual naturalized runoff')
ax2_2.plot(guide_infl_obs.index, guide_infl_obs['Ratio'], color=ratio_observed_runoff_color, label='Ratio of annual observed runoff')
ax2_2.set_ylabel('Ratio (%)', fontsize=ftsize)

# 子图3：循化影响前期径流
axes[1][0].bar(xunhua_preinfl.index, xunhua_preinfl['runoff(10^8m^3)'], color=natural_runoff_color)
axes[1][0].set_xlabel('Month', fontsize=ftsize)
axes[1][0].set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
axes[1][0].set_title('(c) Pre-influenced Period at Xunhua', fontsize=ftsize)

# 创建第二个y轴
ax2_3 = axes[1][0].twinx()
ax2_3.plot(xunhua_preinfl.index, xunhua_preinfl['Ratio'], color=ratio_natural_runoff_color)
ax2_3.set_ylabel('Ratio (%)', fontsize=ftsize)

# 子图4：循化影响后期径流
axes[1][1].bar(xunhua_infl.index - 0.2, xunhua_infl['runoff(10^8m^3)'], width=0.4, color=naturalized_runoff_color)
axes[1][1].bar(xunhua_infl.index + 0.2, xunhua_infl_obs['runoff(10^8m^3)'], width=0.4, color=observed_runoff_color)
axes[1][1].set_xlabel('Month', fontsize=ftsize)
axes[1][1].set_ylabel('Runoff($10^8m^3$)', fontsize=ftsize)
axes[1][1].set_title('(d) Influenced Period at Xunhua', fontsize=ftsize)

# 创建第二个y轴
ax2_4 = axes[1][1].twinx()
ax2_4.plot(xunhua_infl.index, xunhua_infl['Ratio'], color=rario_naturalized_runoff_color)
ax2_4.plot(xunhua_infl_obs.index, xunhua_infl_obs['Ratio'], color=ratio_observed_runoff_color)
ax2_4.set_ylabel('Ratio (%)', fontsize=ftsize)

# 获取所有图例
handles, labels = [], []

# 获取主坐标轴的图例
for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)

# 获取twinx坐标轴的图例
for ax in [ax2_1, ax2_2, ax2_3, ax2_4]:
    for handle, label in zip(*ax.get_legend_handles_labels()):
        handles.append(handle)
        labels.append(label)



fig.legend(handles, labels, 
           loc='upper center', 
        #    bbox_to_anchor=(0.5, 1.0), 
           ncol=3, 
           fontsize=ftsize, 
           frameon=False)

# 美化
# plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.06, right=0.93,top=0.90, hspace=0.25, wspace=0.38)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/runoff_distribution.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()
