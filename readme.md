# Codes for article "An extension approach for naturalizing streamflow based on long short-term memory by excluding liner related driving factors"

The Pettitt test and Mann-Kendall test programs developed by Md. Manjurul Hussain Shourov were revised by this study to delineate the pre-influence and influence periods. Sincere gratitude is extended to Mr. Shourov.

## Sample data

full_modelling_data.csv - samples for establishing streamflow naturalization model
nature_flow.csv - Evaluated natureal streamflow by the Ministry of Water Resources of the People's Republic of China for validate the model

## mutation detection

HomogeneityTest.py
OrderedClusterAnalysis.py
MannKendallMutation.py
mutation_detect.py

## Multiple linearity reconstruction

VIF_test.py

## sample generation

OneShotSamplesGenerator.py
gen_samples.ipynb

## extension-based flow estimation

Normalizer.py
LSTMRegressor.py
BuildPytorchLSTMRegressorForGuide.ipynb
BuildPytorchLSTMRegressorForXunhua.ipynb

## result visualization

CorrFlowFeaturesMiniMap.py
mutation_detect_add_tnh.py
NaruralVsObservedMiniMap.py
Plot_Abs_Pearson_CC.py
plot_compare_naturali.py
plot_monthly_streafmlow.py
Plot_PACF.py
plot_runoff_distribution_add_tnh.py
plot_runoff_distribution.py
plot_trend.py
PlotCalTestVal.py

# 建模过程

## 1 数据准备+多重共线性重构（PrepareModellingData.ipynb）

## 2 突变点识别（mutation_detect.py）

## 3 模型参数率定验证与天然流量估算

* 扩展路由融合模型CER(CNN),扩展模型E-CNN,路由模型R-CNN模型: BuildCNNRegressor.ipynb
* 扩展路由融合模型CER(BiLSTM),扩展模型E-BiLSTM,路由模型R-BiLSTM: BuildBiLSTMRegressor.ipynb
* 扩展路由融合模型CER(LSTM),扩展模型E-LSTM,路由模型R-LSTM: BuildLSTMRegressor.ipynb
* 扩展路由融合模型CER(XGB),扩展模型E-XGB,路由模型R-XGB: BuildXGBoostRegressor.ipynb
* 扩展路由融合模型CER(MLR),扩展模型E-MLR,路由模型R-MLR: BuildXGBoostRegressor.ipynb

## 4 结果处理（ResultsManagement.ipynb）
