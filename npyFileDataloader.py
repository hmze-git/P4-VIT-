import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2

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

        
        transformed=v2.Compose([
            v2.RandomResizedCrop(size=(224,224),antialias=True),
            v2.ColorJitter(brightness=0.3,contrast=0.2,saturation=0.2),
            v2.RandomHorizontalFlip(0.5)
        ])

        x=transformed(x)
        y=torch.from_numpy(np.asarray(self.yData[idx]))

        return x,y
    def __len__(self):

        
        return self.xData.shape[0]