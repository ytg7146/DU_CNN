import numpy as np
import torch
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self, datashort, datalong,num):
        self.datashort=datashort
        self.datalong=datalong
        self.datarange=len(num)
        self.datarangeshort=datashort[0][0].shape[1]
        self.datarangelong=datalong[0][0].shape[1]
        self.datarange2=datashort[0][0].shape[1]*datalong[0][0].shape[1]
        self.num = num
        self.length = 0

        for i in num:
            self.length += datashort[i][0].shape[1]*datalong[i][0].shape[1]
            

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.datashort[self.num[idx//self.datarange2]][0][:,idx%self.datarange2//self.datarangelong]).float(),\
            torch.from_numpy(self.datalong[self.num[idx//self.datarange2]][0][:,idx%self.datarange2%self.datarangelong]).float()


