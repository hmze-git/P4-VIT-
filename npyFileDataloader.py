import torch
import numpy as np
from torch.utils.data import Dataset

class NumpyLoader(Dataset):
    def __init__(self,xShape,xPath,yShape,yPath):
        super().__init__()
        self.xShape=xShape
        self.xPath=xPath
        self.yShape=yShape
        self.yPath=yPath
        self.xData=np.load(self.xPath,mmap_mode='r')
        self.yData=np.load(self.yPath,mmap_mode='r')


    def __getitem__(self,idx):
        x=torch.from_numpy(np.asarray(self.xData[idx]))
        y=torch.from_numpy(np.asarray(self.yData[idx]))

        return x,y
    def __len__(self):
        return self.xData.shape[0]