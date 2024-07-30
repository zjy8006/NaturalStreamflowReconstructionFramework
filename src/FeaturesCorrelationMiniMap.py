import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 相关系数矩阵
corr = np.array([
    [1, 0.674, 0.587, -0.168, -0.617, -0.639, -0.694],
    [0.674, 1, 0.509, -0.271, -0.559, -0.538, -0.583],
    [0.587, 0.509, 1, -0.345, -0.488, -0.493, -0.511],
    [-0.168, -0.271, -0.345, 1, 0.229, 0.196, 0.198],
    [-0.617, -0.559, -0.488, 0.229, 1, 0.686, 0.7],
    [-0.639, -0.538, -0.493, 0.196, 0.686, 1, 0.75],
    [-0.694, -0.583, -0.511, 0.198, 0.7, 0.75, 1]
])

corr = np.round(corr, 1)

# 设置变量名
# labels = ['v10', 'snowc', 'sd', 'ssro', 'P208_gd', 'P820_gd', 'P820_gh']
labels = ['$V_1$', '$V_2$', '$V_3$', '$V_4$', '$V_5$', '$V_6$', '$V_7$']

# 创建热图
plt.figure(figsize=(2.5, 1.8))
ftsize=8
ax = sns.heatmap(corr, annot=True, cmap='coolwarm',xticklabels=labels, yticklabels=labels,annot_kws={'size': 8})
# plt.title('Correlation Matrix Heatmap')
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=ftsize) # 设置图例刻度的字体大小

plt.subplots_adjust(left=0.08, bottom=0.15, right=0.99,top=0.98, hspace=0.4, wspace=0.25)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/Xunhua_heatmap_mini.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')
plt.show()
