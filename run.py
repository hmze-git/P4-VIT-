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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt
device=None
if torch.cuda.is_available:
  device=torch.device('cuda')

def setSeed():
  seed=67

  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)



setSeed()

NpFile=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\InputFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\OutputTags.npy")
NPFileTest=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\validationFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Honours Project\Pipeline1TFlow\validationTags.npy")

smallTest=Subset(NpFile,list(range(12)))
smallLoader=DataLoader(smallTest,batch_size=1,shuffle=True)

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

learningRate=0.001



lossFunction=nn.CrossEntropyLoss()
validationLossFunction=nn.CrossEntropyLoss()
adamOptimiser=torch.optim.Adam(params=FullModel.parameters(),lr=learningRate)

#tr step lr reduce learn rate by 10X every 10 epochs
#if this does not work try ReduceLRONPateu for when valid accuracy taps out

stepLearnDecay=torch.optim.lr_scheduler.StepLR(adamOptimiser,5,0.1)
stepLearnDecay.optimizer.zero_grad

metric=classification.Accuracy(task='multiclass',num_classes=3)
testMetric=classification.Accuracy(task='multiclass',num_classes=3)
metric=metric.to(device)
testMetric=testMetric.to(device)

if device is not None:
  FullModel=FullModel.to(device)




def trainStep(model,dataLoader,testLoader,metric,testMetric,lossFunction,testLossFunction,optimiser,epoch,lossArr,valLossArr,accArr,valAccArr):
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
  lossArr.append(normLoss)
  valLossArr.append(validNormLoss)
  accArr.append(epochAccuracy)
  valAccArr.append(epochValidationAccuary)

    

   
def trainingLoop(epochs,model,dataLoad,testDataLoad,lossFN,testLossFn,Optimiser,lrDecay,metric,testMetric):

  lossArr=[]
  valLossArr=[]
  accArr=[]
  valAcc=[]

  bestValAcc=0.0
  for e in range(epochs):
      
      trainStep(model,dataLoad,testDataLoad,metric,testMetric,lossFN,testLossFn,Optimiser,e,lossArr,valLossArr,accArr,valAcc)

      #get the very last accuracy and see if higher than best
      mrValAcc=valAcc[-1]


      #Basic checkpoint hist
      if mrValAcc>bestValAcc:
         bestValAcc=mrValAcc
         saveModel(model,e,Optimiser,valLossArr[-1],lossArr[-1])



      metric.reset()
      testMetric.reset()
      lrDecay.step()

  savedModelDict=torch.load('savedModel.tar')

  model.load_state_dict(savedModelDict['modelStateDict'])  
  model.eval()
  with torch.inference_mode():

    yPredict=[] 
    yTrue=[]
    for batch,(x,y) in enumerate(testLoader):
        validInput=x.to(device)
        validLabel=y.to(device)

        validationPreds=model(validInput)
        validIndex=torch.argmax(validationPreds)
        yPredict.append(validIndex)
        yTrue.append(y)

    confMatrix=confusion_matrix(yTrue,yPredict)
    disp=ConfusionMatrixDisplay(confusion_matrix=confMatrix,display_labels=[0,1,2])
    disp.plot(cmap='Blues')
    plt.savefig('confMatrix.png')
    plt.show()
    plt.clf()
    #accuracy and loss plotts
    plt.plot(accArr)
    plt.plot(valAcc)
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','val'])
    plt.savefig('NormVsValACc.png')
    plt.show()
    plt.clf()

    plt.plot(lossArr)
    plt.plot(valLossArr)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'])
    plt.show()
    plt.savefig('NormVsValLoss.png')



    
def saveModel(model,epoch,optimiser,vLoss,loss):

   torch.save({
      'epoch':epoch,
      'modelStateDict':model.state_dict(),
      'optimiserStateDict':optimiser.state_dict(),
      'valLoss':vLoss,
      'loss':loss
   },'savedModel.tar')
   

  


trainingLoop(5,FullModel,trainLoader,testLoader,lossFunction,validationLossFunction,adamOptimiser,stepLearnDecay,metric,testMetric)