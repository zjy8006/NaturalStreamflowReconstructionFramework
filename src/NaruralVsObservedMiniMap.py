import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_excel("H:/连亚妮论文数据/第三章/循化站多种方法还原结果对比.xls",sheet_name='Sheet1')

df['YQ'] = df['年份'].astype(str) + '-' + df['季节'].astype(str)

# 将新创建的 'Year-Quarter' 列设置为索引
df.set_index('YQ', inplace=True)

obs = df['Natural']
lstm = df['VIF-LSTM']

print(obs)
plt.figure(figsize=(4.5, 1.8))
# sns.barplot(y=series.index, x=series.values, palette="coolwarm")  # 使用coolwarm色板
# sns.barplot(x=series.index, y=series.values, palette="coolwarm")  # 使用coolwarm色
ax1 =sns.lineplot(x=obs.index, y=obs.values,label='Natural streamflow')
ax2 = sns.lineplot(x=lstm.index, y=lstm.values,label='Naturalized streamflow byLSTM')

ax1.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
ax2.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')

for label in ax1.get_yticklabels():
    label.set_fontsize(8)

for label in ax2.get_yticklabels():
    label.set_fontsize(8)

# 添加标题和标签
# plt.title('Feature Pearson Correlation Coefficients', fontsize=16)
plt.xlabel('Date', fontsize=8)
plt.ylabel('Streamflow($m^3/s$)', fontsize=8)


# 优化布局和显示图形
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
tick_positions = np.arange(0, len(obs.index), 5)  # 每5个索引一个标签
tick_labels = [obs.index[i] for i in tick_positions]  # 从标签列表中选取这些标签
plt.xticks(tick_positions, tick_labels, rotation=45)  # 设置x轴的标签位置和标签内容
plt.xticks(rotation=45) 
plt.legend(shadow=False,frameon=False,fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
# plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.25, right=0.99,top=0.93, hspace=0.4, wspace=0.25)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/Xunhua_naturalization_mini.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')
plt.show()
