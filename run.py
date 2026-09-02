import torch
from torch.utils.data import DataLoader
from torch import nn
import npyFileDataloader

from transformers import ViTForImageClassification
from torch.utils.data import Subset
import torchinfo
import torchmetrics
from torchmetrics import classification
from ViT import SkinCancerLSTMViT

device=None
if torch.cuda.is_available:
  device=torch.device('cuda')


NpFile=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\InputFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\OutputTags.npy")
NPFileTest=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\validationFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\validationTags.npy")
#smallTest=Subset(NpFile,list(range(12)))
#smallLoader=DataLoader(smallTest,batch_size=4,shuffle=True)

trainLoader=DataLoader(NpFile,batch_size=4,shuffle=True)
testLoader=DataLoader(NPFileTest,batch_size=4) # dont shuffle so that when testing it gets items in same order so it wont fluctuate based on what was given first

modelName="Anwarkh1/Skin_Cancer-Image_Classification"

preTrainedViT=ViTForImageClassification.from_pretrained(modelName)

inputDim=768
hiddenSize=64
for p in preTrainedViT.vit.parameters():
  p.requires_grad = False

preTrainedViT.classifier=nn.Identity()
preTrainedViT.eval()

FullModel=SkinCancerLSTMViT(preTrainedViT,hiddenSize,inputDim,True,3)

learningRate=0.0005



lossFunction=nn.CrossEntropyLoss()
validationLossFunction=nn.CrossEntropyLoss()
adamOptimiser=torch.optim.Adam(params=FullModel.parameters(),lr=learningRate)
metric=classification.Accuracy(task='multiclass',num_classes=3)
testMetric=classification.Accuracy(task='multiclass',num_classes=3)
metric=metric.to(device)
testMetric=testMetric.to(device)

if device is not None:
  FullModel=FullModel.to(device)



def trainStep(model,dataLoader,testLoader,metric,testMetric,lossFunction,testLossFunction,optimiser,epoch):
  #put mode; in train mode
  #change to model.eval whne doing validation loss etc
  model.train()

  accumulatedLoss=0.0
  validationLoss=0.0
  for batch,(x,y) in enumerate(dataLoader):
 
    xInput=x.to(device)
    yLabel=y.to(device)

    yPredictions=model(xInput)


    metric.update(yPredictions,yLabel)

    #Ypredictions might need squeeze to deal with shape mismatch deal later
    loss=lossFunction(yPredictions,yLabel)
    accumulatedLoss+=loss
    optimiser.zero_grad()

    #BACKPROP THE RROR
    loss.backward()

    optimiser.step()

    #set model to evaluation mode to calc validation loss and accuracy
  model.eval()
  with torch.inference_mode():
    
    for batch,(x,y) in enumerate(testLoader):
        validInput=x.to(device)
        validLabel=y.to(device)

        validationPreds=model(validInput)
        testMetric.update(validationPreds,validLabel)
        validLoss=testLossFunction(validationPreds,validLabel)
        validationLoss+=validLoss


  epochAccuracy=metric.compute()
  epochValidationAccuary=testMetric.compute()
  normLoss=accumulatedLoss/len(dataLoader)
  validNormLoss=validationLoss/len(testLoader)

  print(f"Epoch {epoch} | Loss {normLoss} | Validation Loss {validNormLoss} | Accuracy {epochAccuracy} | Valid Accuracy {epochValidationAccuary}")

    

   
def trainingLoop(epochs,model,dataLoad,testDataLoad,lossFN,testLossFn,Optimiser,metric,testMetric):
 
  for e in range(epochs):
      trainStep(model,dataLoad,testDataLoad,metric,testMetric,lossFN,testLossFn,Optimiser,e)
      metric.reset()
      testMetric.reset()


trainingLoop(5,FullModel,trainLoader,testLoader,lossFunction,validationLossFunction,adamOptimiser,metric,testMetric)