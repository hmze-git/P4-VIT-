import torch
import numpy as np
from torch.utils.data import Dataset

class NumpyLoader(Dataset):
    def __init__(self,xPath,yPath):
        super().__init__()
        self.xPath=xPath
        self.yPath=yPath
        self.xData=np.load(self.xPath,mmap_mode='r')
        self.yData=np.load(self.yPath,mmap_mode='r')


    def __getitem__(self,idx):
        #Permute so channel comes before the other 2 otherwise it fails with the ViT 
        #maybe this will break the batching or vidoes idk check later 
        x=torch.from_numpy(np.asarray(self.xData[idx])).permute(0,3,1,2).contiguous()
        y=torch.from_numpy(np.asarray(self.yData[idx]))

        return x,y
    def __len__(self):

        
        return self.xData.shape[0]