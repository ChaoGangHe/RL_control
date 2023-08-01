import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

###   Demo dataset

data_ = [[1, 10, 11, 15, 9, 100],
         [2, 11, 12, 16, 9, 100],
         [3, 12, 13, 17, 9, 100],
         [4, 13, 14, 18, 9, 100],
         [5, 14, 15, 19, 9, 100],
         [6, 15, 16, 10, 9, 100],
         [7, 15, 16, 10, 9, 100],
         [8, 15, 16, 10, 9, 100],
         [9, 15, 16, 10, 9, 100],
         [10, 15, 16, 10, 9, 100]]


###   Demo Dataset class

class DemoDatasetLSTM(Data.Dataset):
    """
        Support class for the loading and batching of sequences of samples
        Args:
            dataset (Tensor): Tensor containing all the samples
            sequence_length (int): length of the analyzed sequence by the LSTM
            transforms (object torchvision.transform): Pytorch's transforms used to process the data
    """

    ##  Constructor
    def __init__(self, dataset, sequence_length=1, transforms=None):
        self.dataset = dataset
        self.seq_len = sequence_length
        self.transforms = transforms

    ##  Override total dataset's length getter
    def __len__(self):
        return self.dataset.__len__()

    ##  Override single items' getter
    def __getitem__(self, idx):#batchsize
        if idx + self.seq_len > self.__len__():
            if self.transforms is not None:
                item = torch.zeros(self.seq_len, self.dataset[0].__len__())
                item[:self.__len__() - idx] = self.transforms(self.dataset[idx:])
                return item, item
            else:
                item = []
                item[:self.__len__() - idx] = self.dataset[idx:]
                return item, item
        else:
            if self.transforms is not None:
                return self.transforms(self.dataset[idx:idx + self.seq_len]), self.transforms(
                    self.dataset[idx:idx + self.seq_len])
            else:
                return self.dataset[idx:idx + self.seq_len], self.dataset[idx:idx + self.seq_len]


###   Helper for transforming the data from a list to Tensor

def listToTensor(list):
    tensor = torch.empty(list.__len__(), list[0].__len__())
    for i in range(list.__len__()):
        tensor[i, :] = torch.FloatTensor(list[i])
    return tensor


###   Dataloader instantiation

# Parameters
seq_len = 3
batch_size = 2
data_transform = transforms.Lambda(lambda x: listToTensor(x))

dataset = DemoDatasetLSTM(data_, seq_len, transforms=data_transform)
print(torch.tensor(data_).shape)
data_loader = Data.DataLoader(dataset, batch_size, shuffle=False)

for data in data_loader:
    x, _ = data
    print(torch.tensor(x).shape)
    print(x)
    print('\n')