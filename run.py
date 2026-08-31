import torch
from torch.utils.data import DataLoader
import npyFileDataloader



NpFile=npyFileDataloader.NumpyLoader((224,224,3),r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\InputFiltered.npy",(1050,1),r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\OutputTags.npy")

trainLoader=DataLoader(NpFile,batch_size=1,shuffle=True)

for vid, label in trainLoader:
    if label.item()!=0:
        print(f"Vid shape {vid.shape}")
        print(f"Label {label.item()}")