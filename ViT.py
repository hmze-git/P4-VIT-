import torch
from torch import nn
import torch
from torch import nn



class SkinCancerLSTMViT(nn.Module):

    def __init__(self,ViT,hiddenDim,inputDim,batchFirst,numClasses):
        super().__init__()
        self.skinViT=ViT
        self.InputDropout=nn.Dropout(0.3)
        self.fullConnect=nn.Sequential(
            nn.Linear(in_features=inputDim,out_features=hiddenDim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDim,numClasses)
        )

    def forward(self,x,realFrameLen):


        
        b,f,c,h,w=x.shape
        x=x.view(b*f,c,h,w)


        #Get the actual numbers (vector after running forward prop)
        extractedFeatures=self.skinViT(x).logits
        extractedFeatures=extractedFeatures.view(b,f,-1)
        extractedFeatures=self.InputDropout(extractedFeatures)

        realFrameLen=realFrameLen.to(extractedFeatures.device)
 
        tfMask=torch.arange(f,device=extractedFeatures.device).unsqueeze(0)  # go from 1d vector to tensor (1,f)
        realFrameLen=realFrameLen.unsqueeze(1) # tensor of (batch,1) obtained mismatch casues stretch
        tfMask=tfMask<realFrameLen
        #tf becomes (b,f)  of true and false

        #add another dim rem that post vit shape is b,f,features this way it matches
        #Brodcasting applied makes features match extracted features
        extractedFeatures=extractedFeatures*tfMask.unsqueeze(-1) 

        #sum features along frame dim and divide by real frame len get average "feature" representation for a video
        #might be cause of crash test later
        extractedFeatures=extractedFeatures.sum(1)/realFrameLen




        #out, outLen = nn.utils.rnn.pad_packed_sequence(packedOut, batch_first=True)
        #try using hn[-1] for last time step
        out=self.fullConnect(extractedFeatures)

        return out
