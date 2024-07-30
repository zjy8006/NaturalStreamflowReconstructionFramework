import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 第一个相关系数矩阵数据
data1 = np.array([
    [1.0, 0.674, 0.587, -0.168, -0.617, -0.639, -0.694],
    [0.674, 1.0, 0.509, -0.271, -0.559, -0.538, -0.583],
    [0.587, 0.509, 1.0, -0.345, -0.488, -0.493, -0.511],
    [-0.168, -0.271, -0.345, 1.0, 0.229, 0.196, 0.198],
    [-0.617, -0.559, -0.488, 0.229, 1.0, 0.686, 0.7],
    [-0.639, -0.538, -0.493, 0.196, 0.686, 1.0, 0.75],
    [-0.694, -0.583, -0.511, 0.198, 0.7, 0.75, 1.0]
])

# 第二个相关系数矩阵数据
data2 = np.array([
    [1.0, 0.595, -0.164, -0.464, -0.492, -0.621, -0.645, -0.694],
    [0.595, 1.0, -0.337, -0.053, -0.118, -0.493, -0.498, -0.514],
    [-0.164, -0.337, 1.0, -0.192, -0.131, 0.231, 0.201, 0.194],
    [-0.464, -0.053, -0.192, 1.0, 0.735, 0.359, 0.35, 0.364],
    [-0.492, -0.118, -0.131, 0.735, 1.0, 0.27, 0.238, 0.281],
    [-0.621, -0.493, 0.231, 0.359, 0.27, 1.0, 0.688, 0.688],
    [-0.645, -0.498, 0.201, 0.35, 0.238, 0.688, 1.0, 0.757],
    [-0.694, -0.514, 0.194, 0.364, 0.281, 0.688, 0.757, 1.0]
])

labels1 = ['v10', 'snowc', 'sd', 'ssro', 'P208_GD', 'P820_GD', 'P820_GH']
labels2 = ['v10', 'sd', 'ssro', 'sro', 'smlt', 'P208_GD', 'P820_GD', 'P820_GH']

df1 = pd.DataFrame(data1, index=labels1, columns=labels1)
df2 = pd.DataFrame(data2, index=labels2, columns=labels2)


# 使用seaborn绘制热力图
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7.48, 5), gridspec_kw={'width_ratios': [7, 8]})
cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.05])  # 调整色带的位置和方向

sns.heatmap(df1, annot=True, fmt=".2f", ax=axes[0], cmap='coolwarm', cbar=False)
axes[0].set_title('(a) Guide')

sns.heatmap(df2, annot=True, fmt=".2f", ax=axes[1], cmap='coolwarm', cbar=True, cbar_ax=cbar_ax, cbar_kws={'orientation': 'horizontal'})
axes[1].set_title('(b) Xunhua')

tick_positions1 = np.arange(0, len(df1.index))  # 每5个索引一个标签 # 从标签列表中选取这些标签
axes[0].set_xticks(tick_positions1+0.5,labels1,rotation=45)
axes[0].set_yticks(tick_positions1+0.5,labels1,rotation=45)

tick_positions2 = np.arange(0, len(df2.index))  # 每5个索引一个标签 # 从标签列表中选取这些标签
axes[1].set_xticks(tick_positions2+0.5,labels2,rotation=45)
axes[1].set_yticks(tick_positions2+0.5,labels2,rotation=45)

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.99,top=0.95, hspace=0.3, wspace=0.3)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/selected_features.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')


plt.show()
