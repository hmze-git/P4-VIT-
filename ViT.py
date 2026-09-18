import torch
from torch import nn
import torch
from torch import nn



class SkinCancerLSTMViT(nn.Module):

    def __init__(self,ViT,hiddenDim,inputDim,batchFirst,numClasses):
        super().__init__()
        self.skinViT=ViT
        self.InputDropout=nn.Dropout(0.3)
        self.Lstm= nn.LSTM(input_size=inputDim,hidden_size=hiddenDim,batch_first=batchFirst)
        #trying to go from the LSTM straight to output head see if that kills overfitting and lets it learn more
        self.fullConnect=nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hiddenDim,numClasses)
        )

    def forward(self,x,realFrameLen):


        b,f,c,h,w=x.shape
        x=x.view(b*f,c,h,w)


        #Get the actual numbers (vector after running forward prop)
        extractedFeatures=self.skinViT(x).logits
        extractedFeatures=extractedFeatures.view(b,f,-1)

        extractedFeatures=self.InputDropout(extractedFeatures)
        #Real fram length padding handler
        packed=nn.utils.rnn.pack_padded_sequence(extractedFeatures,realFrameLen.cpu(),True,False)


        #when usign pack padding it returns pack padded obj so unwrap that layer first to get usual shape
        # then use the final frame as you wish
        packedOut,(hn,cn)=self.Lstm(packed)

        #out, outLen = nn.utils.rnn.pad_packed_sequence(packedOut, batch_first=True)
        #try using hn[-1] for last time step
        out=self.fullConnect(hn[-1])

        return out
