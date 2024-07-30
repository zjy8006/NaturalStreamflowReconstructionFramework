import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
import matplotlib.colors as mcolors

# 生成随机时间序列数据
np.random.seed(0)
data = np.random.randn(100)

data = pd.read_csv('data/guide_hy_1957_1985vif_pearson.csv',index_col='date',parse_dates=True)
data = data['ssro']

# 计算PACF
pacf_values = pacf(data, nlags=30)

# 置信区间阈值（对于大样本近似）
conf_interval = 1.96 / np.sqrt(len(data))

# 绘制PACF图
fig, ax = plt.subplots(figsize=(7.48,3))
# fig, ax = plt.subplots(figsize=(3.54,1.5))

ftsize = 10
# 设置颜色映射和归一化
cmap = plt.get_cmap('coolwarm')
norm = mcolors.Normalize(vmin=-1, vmax=1)

# 绘制PACF值，颜色基于PACF值
bars = ax.bar(range(len(pacf_values)), pacf_values, color=[cmap(norm(value)) for value in pacf_values])

# 绘制置信区间线
ax.axhline(y=conf_interval, linestyle='--', color='gray')
ax.axhline(y=-conf_interval, linestyle='--', color='gray')
ax.axhline(y=0, linestyle='-', color='black',linewidth=0.3)

# 添加色带
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.colorbar(sm, ax=ax, orientation='vertical')

# 标题和标签
ax.set_title('Partial Autocorrelation Function (PACF)', fontsize=ftsize)
ax.set_xlabel('Lag', fontsize=ftsize)
ax.set_ylabel('PACF', fontsize=ftsize)

plt.tight_layout()
# plt.subplots_adjust(left=0.1, bottom=0.3, right=0.99,top=0.95, hspace=0.3, wspace=0.3)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/PACF.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')


# 显示图表
plt.show()
