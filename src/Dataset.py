
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class CustomMultiOutDatasets(Dataset):
    def __init__(self,features:pd.DataFrame,targets:pd.DataFrame)->None:
        self.X = torch.tensor(features.values,dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(targets.values,dtype=torch.float32).to(DEVICE)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]

class CustomDatasets(Dataset):
    def __init__(self,dataframe:pd.DataFrame,target:str):
        self.dataframe = dataframe
        self.target = target
        self.features = list(self.dataframe.columns.difference([self.target]))
        self.X = torch.tensor(self.dataframe[self.features].values,dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(self.dataframe[self.target].values,dtype=torch.float32).to(DEVICE).view(-1,1)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
    
    
class SimulatedSequenceDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, target:str, features:list, sequence_length:int=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()[self.sequence_length:]
        self.X = torch.tensor(dataframe[features].values).float()
        # print('self.features:',self.features)
        # print('self.X',self.X)
        # print('self.y',self.y)

    def __len__(self):
        return self.X.shape[0]-self.sequence_length

    def __getitem__(self, i): 
        i = i + self.sequence_length  # 调整索引使得预测结果与训练结果对齐
        i_start = i - self.sequence_length
        x = self.X[i_start:(i + 1), :]

        # if i >= self.sequence_length - 1: #!drop
        #     i_start = i - self.sequence_length + 1
        #     x = self.X[i_start:(i + 1), :]
        # else:
        #     padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
        #     x = self.X[0:(i + 1), :]
        #     x = torch.cat((padding, x), 0)#!drop

        return x, self.y[i-self.sequence_length]
        # return x.unsqueeze(0), self.y[i]  # 增加一个批次维度


class SequenceDataset(Dataset):
    def __init__(self, dataframe:pd.DataFrame, target:str, features:list, sequence_length:int=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        # print('self.features:',self.features)
        # print('self.X',self.X)
        # print('self.y',self.y)

    def __len__(self):
        return self.X.shape[0]-self.sequence_length

    def __getitem__(self, i): 
        i = i + self.sequence_length  # 调整索引使得预测结果与训练结果对齐
        i_start = i - self.sequence_length
        x = self.X[i_start:i, :]
        return x, self.y[i]
        # return x.unsqueeze(0), self.y[i]  # 增加一个批次维度
    
    
# class SequenceDataset(Dataset): #!drop
#     def __init__(self, dataframe:pd.DataFrame, target:str, features:list, sequence_length:int=5,lead:int=1):
#         self.features = features
#         self.target = target
#         self.sequence_length = sequence_length
#         self.lead = lead
#         self.y = torch.tensor(dataframe[target].values).float()
#         self.X = torch.tensor(dataframe[features].values).float()

#     def __len__(self):
#         return self.X.shape[0]-self.sequence_length - self.lead + 1

#     def __getitem__(self, i): 
#         i = i + self.sequence_length - 1  # 调整索引使得预测结果与训练结果对齐
#         if i >= self.sequence_length - 1:
#             i_start = i - self.sequence_length + 1
#             x = self.X[i_start:(i + 1), :]
#         else:
#             padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
#             x = self.X[0:(i + 1), :]
#             x = torch.cat((padding, x), 0)
#         y_index = i + self.lead
#         return x, self.y[y_index]
    


if __name__ == "__main__":
    df = pd.read_csv('data/data8006.csv', index_col=['time'], parse_dates=['time'])
    # print(df)
    df = df.drop(['index'],axis=1)

    # dataset = SequenceDatasetTest(
    #     dataframe=df,
    #     target='R',
    #     features=list(df.columns.difference(['R'])),
    #     sequence_length=4
    # )
    # i=5
    # X, y = dataset[i]
    # print(X)

    # dataset = SequenceDataset(
    #     dataframe=df,
    #     target='R',
    #     features=list(df.columns.difference(['R'])),
    #     input_sequence_length=5,
    #     lead=2
    # )

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # X, y = next(iter(dataloader))

    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)

    # # for x, y in dataloader:
    # #     print("输入：", x)
    # #     print("输出：", y)

    # import sys
    # sys.path.append('.')
    # from HydroPy.Preprocessing.OneShotSamplesGenerator import gen_one_out_samples
    # samples,target,features = gen_one_out_samples(
    #     timeseries=df,
    #     target_column='R',
    #     lead=1,
    #     lag=5,
    #     mode='simulate'
    # )

    # # print(samples)

    # dataset = SequenceDatasetFromSamples(
    #     dataframe=samples,
    #     target=target,
    #     features=list(samples.columns.difference([target])),
    # )

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    # X, y = next(iter(dataloader))

    # print("Features shape:", X.shape)
    # print("Target shape:", y.shape)

    # for x, y in dataloader:
    #     print("输入：", x)
    #     print("输出：", y)


    
