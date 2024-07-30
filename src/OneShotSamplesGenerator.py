import pandas as pd

def gen_one_out_samples(timeseries:pd.DataFrame, target_column:str,lead:int,lag:int,lags_dict:dict=None,mode:str='forecast'):
    """ Generate one-shot samples for one-output target.

    Parameters:
    -----------
    * `timeseries`: pd.DataFrame
        Timeseries.
    * `target_column`: str
        Target column name.
    * `lead`: int
        The forecast period.
    * `lag`: int
        The laged period.
    * `lags_dict`: dict
        Lags dict. If not specified, will be generated automatically.
    * `mode`: str
        The mode of generating samples. 'simulate' or 'forecast'.
        The simulate mode will consider the lagged period as well as the forecast period as input,
        while the forecast mode will only consider the lagged period as input.

    Returns:
    -----------
    * `results=(samples,target,features)`: tuple
        Samples: pd.DataFrame
        target: str list
        features: str list
    """    
    target = timeseries.pop(target_column)

    if lags_dict is None and lag is not None:
        lags_dict = {col: lag for col in timeseries.columns}

    max_lag = max(lags_dict.values())
    features = list(timeseries.columns)
    features_num = len(features)
    data_size = timeseries.shape[0]
    samples_size = data_size-max_lag
    samples = pd.DataFrame()
    if mode == 'forecast':
        oness_ = pd.DataFrame()
        for i in range(features_num):
            one_in = (timeseries[features[i]]).values
            lag = lags_dict[features[i]]
            oness = pd.DataFrame()
            for j in range(lag): # lag = 2;j=0,1,datasize = 12
                x = pd.DataFrame(one_in[j:data_size-(lag-j)],columns=[
                    features[i]+'_{t-'+ str(lag-j-1)+'}' if lag-j-1!=0 else features[i]+'_{t}'
                ])
                oness = pd.concat([oness, x], axis=1, sort=False)
            oness_ = pd.concat([oness_, oness], axis=1, sort=False)
        oness_ = oness_.iloc[oness_.shape[0]-samples_size:]
        oness_ = oness_.reset_index(drop=True)
        samples = pd.concat([samples, oness_], axis=1, sort=False)
        target = target[max_lag+lead-1:]
        time_index = target.index
        target = pd.DataFrame(target.values,columns=[target_column+'_{t+'+str(lead)+'}'])
        samples = samples[:samples.shape[0]-(lead-1)]
        samples = pd.concat([samples, target], axis=1)
        samples = samples.set_index(time_index,drop=True)
        target_name = target.columns[0]
        features_names = list(samples.columns.difference([target_name]))
    
        
    elif mode == 'simulate':
        oness_ = pd.DataFrame()
        for i in range(features_num):
            one_in = (timeseries[features[i]]).values
            lag = lags_dict[features[i]]
            oness = pd.DataFrame()
            for j in range(lag): # lag = 2;j=0,1,datasize = 12
                x = pd.DataFrame(one_in[j:data_size-(lag+lead-j-1)],columns=[
                    features[i]+'_{t-'+ str(lag-j-1)+'}' if lag-j-1!=0 else features[i]+'_{t}'
                ])
                oness = pd.concat([oness, x], axis=1, sort=False)
            for j in range(lag,lag+lead): # lag+lead = 2+3;j=2,3,4
                x_ = pd.DataFrame(one_in[j:data_size-(lag+lead-j)+1],columns=[features[i]+'_{t+'+ str(j-lag+1)+'}'])
                oness = pd.concat([oness, x_], axis=1, sort=False)
            oness_ = pd.concat([oness_, oness], axis=1, sort=False)
        oness_ = oness_.reset_index(drop=True)
        samples = pd.concat([samples, oness_], axis=1, sort=False)
        target = target[max_lag+lead-1:]
        time_index = target.index
        target = pd.DataFrame(target.values,columns=[target_column+'_{t+'+str(lead)+'}'])
        samples = pd.concat([samples, target], axis=1)
        samples = samples.set_index(time_index,drop=True)
        target_name = target.columns[0]
        features_names = list(samples.columns.difference([target_name]))

    return samples,target_name,features_names

def gen_multi_output_samples(timeseries:pd.DataFrame,target_column:str,lead:int,lag:int):
    """ Generate multi-output target samples.

    Parameters:
    -----------
    * `timeseries`: pd.DataFrame
        Timeseries.
    * `target_column`: str
        Target column name.
    * `lead`: int
        The forecast period.
    * `lag`: int
        The laged period.
    Returns:
    -----------
    * `results=(features_samples,target_samples)`: tuple
        features_samples: pd.DataFrame
        target_samples: pd.DataFrame
    """
    target = timeseries.pop(target_column)
    targets = pd.DataFrame()
    index = target.iloc[lag:target.shape[0]-(lead-1)].index
    for i in range(lead):
        targets[target_column+'_t+'+str(i+1)] = target.iloc[lag+i:target.shape[0]-(lead-i-1)].values
    target_samples = targets.set_index(index)


    # Generate features samples
    features = timeseries.columns.tolist()
    print('features: ', features)
    feature_samples = pd.DataFrame()
    for feature in features:
        for i in range(lag):
            feature_samples[feature+'_t-'+str(lag-i)] = timeseries[feature].iloc[i:timeseries.shape[0]-(lag+lead-i-1)].values
            data = timeseries[feature].iloc[i:timeseries.shape[0]-(lag+lead-i)].values
            # print(data)
        for i in range(lead):
            feature_samples[feature+'_t+'+str(i+1)] = timeseries[feature].iloc[lag+i:timeseries.shape[0]-(lead-i-1)].values
    feature_samples = feature_samples.set_index(index)
    # print('feature_samples: ', feature_samples)
    # print('target_samples: ', target_samples)
    return feature_samples, target_samples


    



if __name__ == '__main__':
    df = pd.read_csv('data/data8006.csv', index_col=['time'], parse_dates=['time'])
    # print(df)
    df = df.drop(['index'],axis=1)

    samples,target,freatures = gen_one_out_samples(
        timeseries=df,
        target_column='R',
        lead=3,
        lag=2,
        mode='simulate'
    )

    # print(samples)


    # df = pd.read_csv('data/MonthlyPcPTempFlow_TNH.csv',index_col=['DATE'],parse_dates=['DATE'])
    # features,targets = gen_multi_output_samples(
    #     timeseries=df,
    #     target_column='month_flow',
    #     lead=3,
    #     lag=2,
    # )

    # samples = gen_one_shot_samples(
    #     timeseries=df,
    #     target_column='R',
    #     lead=2,
    #     lag=2,
    # )

    # print(samples)

