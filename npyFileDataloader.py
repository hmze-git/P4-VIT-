
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
import math
class NumpyLoader(Dataset):
    def __init__(self,xPath,yPath,frameLenPAth,frameSample=5,trainMode=True):
        super().__init__()
        self.xPath=xPath
        self.yPath=yPath
        self.fRPath=frameLenPAth
        self.xData=np.load(self.xPath,mmap_mode='r')
        self.yData=np.load(self.yPath,mmap_mode='r')
        self.frData=np.load(self.fRPath,mmap_mode='r')
        self.frameSample=frameSample
        self.trainMode=trainMode
        self.maxWindowCount=int(math.ceil(70/frameSample))
        
        normalise=v2.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        if trainMode:
               self.transformed=v2.Compose([
                        v2.RandomResizedCrop(size=(224,224),antialias=True,scale=(0.6,1.0)),
                        v2.ColorJitter(brightness=0.3,contrast=0.2),
                        v2.RandomHorizontalFlip(0.5),
                        v2.RandomRotation(15),
                        v2.RandAugment(5,5),
                        v2.RandomErasing(0.25),
                        normalise,
                    ])
        else:
             self.transformed=v2.Compose([
                  normalise

             ])


    def __getitem__(self,idx):
        #Permute so channel comes before the other 2 otherwise it fails with the ViT
        #maybe this will break the batching or vidoes idk check later
        
        nReal=min(int(self.frData[idx]),self.xData.shape[1])
        selectedFrames=self.frameSampling(nReal)       
        windowsEnd=len(selectedFrames)
             
        
        tempx=torch.from_numpy(np.asarray(self.xData[idx][selectedFrames])).permute(0,3,1,2).contiguous()

        x=torch.zeros((self.maxWindowCount,*tempx.shape[1:]))

        #give tempx with the padded 0s everything we worked to up to this point
        x[:windowsEnd]=tempx

        x=self.transformed(x)
        y=torch.from_numpy(np.asarray(self.yData[idx]))


        return x,y,windowsEnd
    def __len__(self):
        return self.xData.shape[0]


    def frameSampling(self,fr):
        frameStop=70
        if fr<70:
            frameStop=fr
    
         #70 is max count 
         #Always pick the first frame to start
        startVal=np.arange(0,frameStop,self.frameSample)
        
        arrayRand=[]
        for x in range(len(startVal)):
              if x+1==len(startVal):
                break
              if self.trainMode:
                randInt=np.random.randint(startVal[x],startVal[x+1])
                arrayRand.append(randInt)
              else:
                  randInt=(startVal[x]+startVal[x+1]-1)//2
                  arrayRand.append(randInt)


        return arrayRand