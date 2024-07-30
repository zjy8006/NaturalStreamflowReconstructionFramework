import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建一个Pandas Series对象
data = {
    "RHU_tr": 0.767826, "RHU_gh": 0.733454, "RHU_gd": 0.728700,
    "MIN_T_tr": 0.683784, "MIN_T_gd": 0.674592, "MIN_T_gh": 0.673913,
    "AVG_T_tr": 0.643687, "AVG_T_gh": 0.642017, "AVG_T_gd": 0.640766,
    "P2020_gh": 0.634730, "P2020_tr": 0.634077, "MAX_T_gh": 0.624351,
    "MAX_T_gd": 0.618596, "P2020_gd": 0.613853, "MAX_T_tr": 0.606574,
    "P208_tr": 0.604756, "P820_gh": 0.598272, "P208_gh": 0.588071,
    "P820_tr": 0.587346, "P208_gd": 0.579501, "P820_gd": 0.548378,
    "EVP_gd": 0.323940, "EVP_gh": 0.300505, "EVP_tr": 0.265952,
    "AVG_W_gd": 0.046889, "AVG_W_gh": -0.176168, "AVG_W_tr": -0.218365
}
series = pd.Series(data)

new_index = [f"$V_{{{i}}}$" for i in range(1, len(data) + 1)]
series.index = new_index  # 赋值新的索引

# 使用Seaborn绘制条形图
plt.figure(figsize=(4.5, 1.8))
ftsize = 8
# sns.barplot(y=series.index, x=series.values, palette="coolwarm")  # 使用coolwarm色板
sns.barplot(x=series.index, y=series.values, palette="coolwarm")  # 使用coolwarm色

# 添加标题和标签
# plt.title('Feature Pearson Correlation Coefficients', fontsize=16)
plt.xlabel('', fontsize=ftsize)
plt.ylabel('Correlation', fontsize=ftsize)
plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)

# 优化布局和显示图形
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(rotation=90) 
# plt.tight_layout()
plt.subplots_adjust(left=0.08, bottom=0.2, right=0.99,top=0.98, hspace=0.4, wspace=0.25)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/Xunhua_corrbar_mini.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')
plt.show()