import torch
from torch.utils.data import DataLoader
from torch import nn
import npyFileDataloader

from transformers import ViTForImageClassification
from torch.utils.data import Subset
import torchinfo
from ViT import SkinCancerLSTMViT

device=None
if torch.cuda.is_available:
  device=torch.device('cuda')


NpFile=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\InputFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\OutputTags.npy")

smallTest=Subset(NpFile,list(range(4)))
smallLoader=DataLoader(smallTest,batch_size=1,shuffle=True)

trainLoader=DataLoader(NpFile,batch_size=1,shuffle=True)

modelName="Anwarkh1/Skin_Cancer-Image_Classification"

preTrainedViT=ViTForImageClassification.from_pretrained(modelName)

inputDim=768
hiddenSize=64
for p in preTrainedViT.vit.parameters():
  p.requires_grad = False

preTrainedViT.classifier=nn.Identity()
preTrainedViT.eval()

FullModel=SkinCancerLSTMViT(preTrainedViT,hiddenSize,inputDim,True,3)

if device is not None:
  FullModel=FullModel.to(device)
 

for x,y in iter(smallLoader):

  vid=x.to(device)
  label=y.to(device)
  
  prediction=FullModel(vid)

      
   
