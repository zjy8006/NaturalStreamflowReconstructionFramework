import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sklearn.preprocessing import MinMaxScaler
from Dataset import SimulatedSequenceDataset
from torch.utils.data import Dataset, DataLoader, TensorDataset,Subset
from OneShotSamplesGenerator import gen_one_out_samples
from MultiOutXGBRegressor import load_model as load_xgb_model
from LSTMRegressor import load_best_trial as load_lstm_best_trial
from LSTMRegressor import load_model as load_lstm_model
from LSTMRegressor import DEVICE
from BiLSTMRegressor import load_best_trial as load_bilstm_best_trial
from BiLSTMRegressor import load_model as load_bilstm_model
from CNNRegressor import load_best_trial as load_cnn_best_trial
from CNNRegressor import load_model as load_cnn_model

hydro_stations = [
    'Guide',
    'Xunhua'
]

start_date = '1960-01-01'
end_date = '2019-12-31'

# Create figure with 3x2 subplots
plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(3, 2, figsize=(7.48, 7.2))

for col, hydro_station in enumerate(hydro_stations):
    # Get XGBoost global importance
    df = pd.read_csv(f'data/{hydro_station.lower()}_vif_modeling_data_1960-2019.csv',parse_dates=['date'],index_col='date')
    df = df.rename(columns={'AVG-TEM(C)': 'AVGT'})
    samples,target,features = gen_one_out_samples(df,target_column='flow',mode='simulate',lags_dict=None,lag=12,lead=1)
    
    # Load trained XGBoost model
    model = load_xgb_model(f'models/CER/{hydro_station.lower()}/XGBOOST_lag12/vif_xgb_model.json')

    # Group features by base name (without lag)
    feature_groups = {}
    for feature in features:
        base_name = feature.split('_')[0]  # Get base name before lag
        if base_name not in feature_groups:
            feature_groups[base_name] = []
        feature_groups[base_name].append(feature)

    # Get global feature importance
    global_importance = model.get_score(importance_type='gain')
    global_contributions = {}
    
    # Calculate aggregated importance for feature groups
    for base_name, feature_list in feature_groups.items():
        global_contributions[base_name] = sum(global_importance.get(f, 0) for f in feature_list)

    # Plot XGBoost global importance
    features_list = list(feature_groups.keys())
    global_importance_df = pd.DataFrame({
        'Predictors': features_list,
        'Importance': [global_contributions[f] for f in features_list]
    })
    
    # Sort features by importance and store the order
    global_importance_df = global_importance_df.sort_values('Importance', ascending=True)
    feature_order = global_importance_df['Predictors'].tolist()

    axes[0,col].barh(global_importance_df['Predictors'], global_importance_df['Importance'])
    axes[0,col].set_xlabel('Global Feature Contribution (gain)')
    axes[0,col].set_title(f'Feature contribution estimated by VIF-XGB for {hydro_station}',fontsize=8)

    print("-"*20,hydro_station,"-"*20)
    df = pd.read_csv(f'data/{hydro_station.lower()}_vif_modeling_data_1960-2019.csv',parse_dates=['date'],index_col='date')
    df = df.loc[start_date:end_date]

    target = 'flow'
    features = list(df.columns.difference([target]))
    features = list(df.columns.copy())
    features.remove(target)
    print(features)

    cal = df.loc[:'1982-12-31',:]
    test = df.loc['1982-01-01':'1985-12-31',:]
    pre = df.loc['1985-01-01':,:]

    X_scaler = MinMaxScaler(feature_range=(0,1))
    Y_scaler = MinMaxScaler(feature_range=(0,1))
    X_scaler.fit(cal[features])
    Y_scaler.fit(cal[[target]])

    cal_X = X_scaler.transform(cal[features])
    test_X = X_scaler.transform(test[features])
    pre_X = X_scaler.transform(pre[features])

    cal_y = Y_scaler.transform(cal[[target]])
    test_y = Y_scaler.transform(test[[target]])
    pre_y = Y_scaler.transform(pre[[target]])

    cal = pd.concat([pd.DataFrame(cal_X,columns=features,index=cal.index),pd.DataFrame(cal_y,columns=[target],index=cal.index)],axis=1)
    test = pd.concat([pd.DataFrame(test_X,columns=features,index=test.index),pd.DataFrame(test_y,columns=[target],index=test.index)],axis=1)
    pre = pd.concat([pd.DataFrame(pre_X,columns=features,index=pre.index),pd.DataFrame(pre_y,columns=[target],index=pre.index)],axis=1)

    sequence_length =12
    cal_dataset = SimulatedSequenceDataset(
        dataframe=cal.copy(),
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SimulatedSequenceDataset(
        dataframe=test.copy(),
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    pre_dataset = SimulatedSequenceDataset(
        dataframe=pre.copy(),
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    full_dataset = SimulatedSequenceDataset(
        dataframe=df.copy(),
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    # Get LSTM global importance using Integrated Gradients
    model_path = f'models/CER/{hydro_station.lower()}/LSTM_lag12/'
    best_trial = load_lstm_best_trial(model_path+'best_trial.pickle')
    model1 = load_lstm_model(model_path+'model.pickle').to(DEVICE)
    ig = IntegratedGradients(model1)

    cal_dataloader = DataLoader(cal_dataset)
    aggregated_attributions = []
    for inputs, labels in cal_dataloader:
        attributions = ig.attribute(inputs.to(DEVICE))
        aggregated_attributions.append(attributions.cpu())
    aggregated_attributions = torch.cat(aggregated_attributions, dim=0)
    attributions_np = aggregated_attributions.detach().numpy()
    sum_attributions = np.sum(attributions_np, axis=0)
    sum_attributions = np.sum(sum_attributions, axis=0)

    # Create DataFrame for LSTM attributions using same feature order as XGBoost
    lstm_df = pd.DataFrame({
        'Features': feature_order[::-1],  # Reverse the feature order
        'Attribution Values': [sum_attributions[features_list.index(f)] for f in feature_order[::-1]]  # Reverse the attribution values
    })
    
    sns.barplot(x='Attribution Values', y='Features', data=lstm_df, orient='h', ax=axes[1,col])
    axes[1,col].set_xlabel('Aggregated Attribution Values')
    axes[1,col].set_title(f'Feature contribution estimated by VIF-LSTM for {hydro_station}',fontsize=8)

    # Get BiLSTM global importance using Integrated Gradients
    model_path = f'models/CER/{hydro_station.lower()}/BiLSTM_lag12/'
    best_trial = load_bilstm_best_trial(model_path+'best_trial.pickle')
    model2 = load_bilstm_model(model_path+'model.pickle').to(DEVICE)
    ig = IntegratedGradients(model2)

    cal_dataloader = DataLoader(cal_dataset)
    aggregated_attributions = []
    for inputs, labels in cal_dataloader:
        attributions = ig.attribute(inputs.to(DEVICE))
        aggregated_attributions.append(attributions.cpu())
    aggregated_attributions = torch.cat(aggregated_attributions, dim=0)
    attributions_np = aggregated_attributions.detach().numpy()
    sum_attributions = np.sum(attributions_np, axis=0)
    sum_attributions = np.sum(sum_attributions, axis=0)

    # Create DataFrame for BiLSTM attributions
    bilstm_df = pd.DataFrame({
        'Features': feature_order[::-1],
        'Attribution Values': [sum_attributions[features_list.index(f)] for f in feature_order[::-1]]
    })
    
    sns.barplot(x='Attribution Values', y='Features', data=bilstm_df, orient='h', ax=axes[2,col])
    axes[2,col].set_xlabel('Aggregated Attribution Values')
    axes[2,col].set_title(f'Feature contribution estimated by VIF-BiLSTM for {hydro_station}',fontsize=8)

fig.tight_layout()
plt.savefig('figs/global_feature_importance_comparison.tif', format='TIFF', dpi=500, bbox_inches='tight')
plt.savefig('figs/global_feature_importance_comparison.png', format='PNG', dpi=600, bbox_inches='tight')
plt.show()
