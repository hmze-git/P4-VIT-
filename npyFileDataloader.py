
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2

class NumpyLoader(Dataset):
    def __init__(self,xPath,yPath,frameLenPAth,trainMode=True):
        super().__init__()
        self.xPath=xPath
        self.yPath=yPath
        self.fRPath=frameLenPAth
        self.xData=np.load(self.xPath,mmap_mode='r')
        self.yData=np.load(self.yPath,mmap_mode='r')
        self.frData=np.load(self.fRPath,mmap_mode='r')

        normalise=v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        if trainMode:
               self.transformed=v2.Compose([
                        v2.RandomResizedCrop(size=(224,224),antialias=True,scale=(0.6,1.0)),
                        v2.ColorJitter(brightness=0.3,contrast=0.2),
                        v2.RandomHorizontalFlip(0.5),
                        v2.RandomRotation(15),
                        normalise
                    ])
        else:
             self.transformed=v2.Compose([
                  normalise

             ])


    def __getitem__(self,idx):
        #Permute so channel comes before the other 2 otherwise it fails with the ViT
        #maybe this will break the batching or vidoes idk check later
        x=torch.from_numpy(np.asarray(self.xData[idx])).permute(0,3,1,2).contiguous()



        x=self.transformed(x)
        y=torch.from_numpy(np.asarray(self.yData[idx]))
        fr=torch.from_numpy(np.asarray(self.frData[idx]))

        return x,y,fr
    def __len__(self):
        return self.xData.shape[0]