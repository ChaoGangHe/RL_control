import random
import time

import torch
import torch.utils.data as Data
import torch.nn as nn
# dataset = DemoDatasetLSTM(data, seq_len, transforms=data_transform)
# data_loader = Data.DataLoader(dataset, batch_size, shuffle=False)
def DemoDatasetLSTM1(dataset,sequence_lenth):

    #print(torch.tensor(dataset).shape[1])
    batch_size = len(dataset)
    #print(batch_size)
    two_dim = len(dataset[0]) - sequence_lenth + 1
    #print(two_dim)
    three_dim = sequence_lenth
    tensor = torch.empty(batch_size,two_dim,three_dim)
    #print(tensor.shape)
    for i in range(0,batch_size):
        for j in range(0,two_dim):
            for k in range(0,three_dim):
                tensor[i][j][k] = dataset[i][j+k]
    return tensor
###   Dataloader instantiation
tensor = torch.empty(128,400)
for i in range(0,128):
    for j in range(0,400):
        tensor[i][j] = i+j


sequence_len = 50
inputs_len = 400 - sequence_len + 1
hidden_size = 16
conv = nn.Conv1d(in_channels=sequence_len,out_channels=256,kernel_size=3)
ReLU = nn.ReLU()
MaxPool1d = nn.MaxPool1d(kernel_size=3,stride=1)
lstm = nn.LSTM(input_size=256,hidden_size=hidden_size,num_layers=2,batch_first=True)
fc = nn.Sequential(
            nn.Linear(347*hidden_size,1024),
            nn.Linear(1024,256),
            nn.Linear(256,64),
            nn.Linear(64,4)
        )

tensor = DemoDatasetLSTM1(tensor,sequence_len)

#print(tensor.shape)
#torch.Size([10, 4, 5])
x = tensor.permute(0, 2, 1)
#print(x.shape)
#torch.Size([10, 5, 4])
x = conv(x)
#print(x.shape)
#torch.Size([1, 256, 349])
x = ReLU(x)
#torch.Size([1, 256, 349])
x = MaxPool1d(x)
#torch.Size([1, 256, 347])
x = x.permute(0, 2, 1)
#torch.Size([1, 347, 256])
x, _ = lstm(x)
#print(x.shape)
#torch.Size([1, 347, 16])
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
#print(x.shape)


# print(tensor)
# tensor = tensor[:, -1, :]
# print(tensor.shape)
# print(tensor)
# tensor = tensor.flatten()
# print(tensor.shape)