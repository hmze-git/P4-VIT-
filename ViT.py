import torch
from torch import nn


class SkinCancerLSTMViT(nn.Module):

    def __init__(self,ViT,hiddenDim,inputDim,batchFirst,numClasses):
        super().__init__()
        self.skinViT=ViT
        self.Lstm= nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first=batchFirst)
        self.fullConnect=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hiddenDim,32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32,numClasses)
        )

    def forward(self,x):

  
        b,f,c,h,w=x.shape
        x=x.view(b*f,c,h,w)
        #Get the actual numbers (vector after running forward prop)
        extractedFeatures=self.skinViT(x).logits
        extractedFeatures=extractedFeatures.view(b,f,-1)

        out,(hn,cn)=self.Lstm(extractedFeatures)

        out=self.fullConnect(out[:,-1,:])

        return out
