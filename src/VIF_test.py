from numpy.core.numeric import full
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import copy

def compute_VIF(X):
    vif_data = pd.DataFrame()
    vif_data['feature']=X.columns
    vif_data['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
    vif_data =vif_data.sort_values(by='VIF',ascending=False,ignore_index=True)
    return vif_data

def recursive_compute_VIF(X_,temp_path):
    full_ = {}
    for feature in X_.columns:
        full_[feature] = []

    def compute(X,df = pd.DataFrame(),index=0):
        vif_data = pd.DataFrame()
        vif_data['feature']=X.columns
        vif_data['VIF']=[variance_inflation_factor(X.values,i) for i in range(len(X.columns))]
        vif_data =vif_data.sort_values(by='VIF',ascending=False,ignore_index=True)
        infos = []
        for i in range(vif_data.shape[0]):
            info = vif_data['feature'][i]+'(%.1E)'%(vif_data['VIF'][i])
            infos.append(info)
        info_df = pd.DataFrame(infos,columns=['fe(VIF),i='+str(index)])
        df = pd.concat([df,info_df],axis=1)
        df.to_csv(temp_path+'/vif.csv')
        info_df.to_csv(temp_path+'/tnh_mete_vif_it'+str(index)+'.csv',index=None)
        index = index+1
        for col in X_.columns:
            if col in vif_data['feature']:
                full_[col].append(vif_data[vif_data['feature']==col]['VIF'])
            else:
                full_[col].append(' ')
        if vif_data['VIF'][0]>5:
            X = X.drop([vif_data['feature'][0]],axis=1)

            print('-'*5+'remove '+vif_data['feature'][0]+'-'*5)
            compute(X,df,index)
        else:
            print(list(vif_data['feature'].values))
            selected_features = list(vif_data['feature'].values)
            return selected_features

    selected_features = compute(X_)
    return selected_features




