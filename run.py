import torch
from torch.utils.data import DataLoader
from torch import nn
import npyFileDataloader
from transformers import ViTForImageClassification
from torch.utils.data import Subset
from torchmetrics import classification
from ViT import SkinCancerLSTMViT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from EarlyStopper import EarlyStopping
import numpy as np
device=None
if torch.cuda.is_available():
  device=torch.device('cuda')

def setSeed():
  seed=67

  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)



setSeed()

NpFile=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\InputFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\OutputTags.npy",r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\RealFrameLen.npy",5,True)
NPFileTest=npyFileDataloader.NumpyLoader(r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\validationFiltered.npy",r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\validationTags.npy",r"C:\Users\Hamzah\Desktop\HYP\Dataset\Working\Grey\ValidationRealFrameLen.npy",5,False)

#smallTest=Subset(NpFile,list(range(12)))
#smallLoader=DataLoader(smallTest,batch_size=2,shuffle=True)
#smallValidTest=Subset(NPFileTest,list(range(12)))
#smallValidLoad=DataLoader(smallValidTest,batch_size=2)

trainLoader=DataLoader(NpFile,batch_size=6,shuffle=True)
testLoader=DataLoader(NPFileTest,batch_size=6) # dont shuffle so that when testing it gets items in same order so it wont fluctuate based on what was given first


#skin cancer link Anwarkh1/Skin_Cancer-Image_Classification


modelName="google/vit-base-patch16-224"

preTrainedViT=ViTForImageClassification.from_pretrained(modelName)

inputDim=768
hiddenSize=64



def unfreezeParts(model,numLayers):

    for p in model.vit.parameters():
         p.requires_grad = False

    if numLayers==0:
       return

    totalLayers = len(model.vit.layers)


    for layer in model.vit.layers[totalLayers-numLayers:totalLayers]:
         layer.requires_grad_(True)




unfreezeParts(preTrainedViT,0)

preTrainedViT.classifier=nn.Identity()


FullModel=SkinCancerLSTMViT(preTrainedViT,hiddenSize,inputDim,True,3)

#Early Stopping Init
earlStop=EarlyStopping(patience=10,delta=0)


#linear scaling abtch size rule
#when batch changes so must lr rule states new LR=lrOLd*(batchSizeNew/batchSizeOld)

learningRate=0.001

lossFunction=nn.CrossEntropyLoss(label_smoothing=0.1)
validationLossFunction=nn.CrossEntropyLoss(label_smoothing=0.1)

otherParams = [p for n, p in FullModel.named_parameters() if not n.startswith("skinViT")]

params = [
    {"params": FullModel.skinViT.parameters(), "lr": learningRate * 0.01, "weight_decay":0.2},
    {"params": otherParams, "lr": learningRate,"weight_decay":0.01},
]




adamOptimiser = torch.optim.AdamW(params=params)

#tr step lr reduce learn rate by 10X every 10 epochs
#if this does not work try ReduceLRONPateu for when valid accuracy taps out


stepLearnDecay=torch.optim.lr_scheduler.ReduceLROnPlateau(adamOptimiser,mode='min',factor=0.5,patience=5,threshold=1e-4,cooldown=0,min_lr=25e-5)


metric=classification.Accuracy(task='multiclass',num_classes=3)
testMetric=classification.Accuracy(task='multiclass',num_classes=3)
testPreicision=classification.MulticlassPrecision(num_classes=3,average=None)
testRecall=classification.MulticlassRecall(num_classes=3,average=None)
metric=metric.to(device)
testMetric=testMetric.to(device)
testPreicision=testPreicision.to(device)
testRecall=testRecall.to(device)

if device is not None:
  FullModel=FullModel.to(device)

  if torch.cuda.device_count()>1:
     FullModel=nn.DataParallel(FullModel)






def trainStep(model,dataLoader,testLoader,metric,testMetric,lossFunction,testLossFunction,optimiser,epoch,lossArr,valLossArr,accArr,valAccArr,testPres,testRec,testpressArr,testRecArr):
  #put mode; in train mode
  #change to model.eval whne doing validation loss etc
  model.train()



  accumulatedLoss=0.0
  validationLoss=0.0
  for batch,(x,y,fr) in enumerate(dataLoader):

    xInput=x.to(device)
    yLabel=y.to(device)
    realFrameLen=fr # do not send to cuda because it will break if you do

    yPredictions=model(xInput,realFrameLen)


    metric.update(yPredictions,yLabel)

    #Ypredictions might need squeeze to deal with shape mismatch deal later
    loss=lossFunction(yPredictions,yLabel)
    accumulatedLoss+=loss.item()
    optimiser.zero_grad()

    #BACKPROP THE RROR
    loss.backward()

    optimiser.step()

    #set model to evaluation mode to calc validation loss and accuracy
  model.eval()
  with torch.no_grad():

    for batch,(x,y,fr) in enumerate(testLoader):
        validInput=x.to(device)
        validLabel=y.to(device)
        realFrameLen=fr

        validationPreds=model(validInput,realFrameLen)
        testMetric.update(validationPreds,validLabel)
        testPres.update(validationPreds,validLabel)
        testRec.update(validationPreds,validLabel)
        validLoss=testLossFunction(validationPreds,validLabel)
        validationLoss+=validLoss.item()


  epochAccuracy=metric.compute()
  epochValidationAccuary=testMetric.compute()

  epochValidationPrecision=testPres.compute()
  epochValidationRecall=testRec.compute()
  normLoss=accumulatedLoss/len(dataLoader)
  validNormLoss=validationLoss/len(testLoader)

  print(f"Epoch {epoch} | Loss {normLoss} | Validation Loss {validNormLoss} | Accuracy {epochAccuracy} | Valid Accuracy {epochValidationAccuary} | Class 1 Precision {epochValidationPrecision[0]} | Class 2 Precision {epochValidationPrecision[1]}| Class 3 Precision {epochValidationPrecision[2]} | Class 1 Recall {epochValidationRecall[0]} | Class 2 Recall {epochValidationRecall[1]} | Class 3 Recall {epochValidationRecall[2]}  ")
  lossArr.append(normLoss)
  valLossArr.append(validNormLoss)
  accArr.append(epochAccuracy.cpu().item())
  valAccArr.append(epochValidationAccuary.cpu().item())
  testpressArr.append(epochValidationPrecision.cpu().numpy())
  testRecArr.append(epochValidationRecall.cpu().numpy())




def trainingLoop(epochs,model,dataLoad,testDataLoad,lossFN,testLossFn,Optimiser,lrDecay,metric,testMetric,testPres,testRec):

  lossArr=[]
  valLossArr=[]
  accArr=[]
  valAcc=[]
  testPresA=[]
  testRecA=[]

  bestValAcc=0.0
  for e in range(epochs):

      trainStep(model,dataLoad,testDataLoad,metric,testMetric,lossFN,testLossFn,Optimiser,e,lossArr,valLossArr,accArr,valAcc,testPres,testRec,testPresA,testRecA)

      #get the very last accuracy and see if higher than best
      mrValAcc=valAcc[-1]


      #Basic checkpoint hist
      if mrValAcc>bestValAcc:
         bestValAcc=mrValAcc
         saveModel(model,e,Optimiser,valLossArr[-1],lossArr[-1])

      # array stores the very last loss value so this should work for early stopping
      if earlStop.stopEarly(valLossArr[-1]):
        print(f"Stopping early at Epoch {e}")
        break


      metric.reset()
      testMetric.reset()
      testPres.reset()
      testRec.reset()
      lrDecay.step(valLossArr[-1])

  savedModelDict=torch.load('savedModel.tar',map_location=device)

  modelToLoad=model.module if isinstance(model,nn.DataParallel) else model
  modelToLoad.load_state_dict(savedModelDict['modelStateDict'])
  model.eval()
  with torch.no_grad():


    presArr=np.asarray(testPresA).T
    recArr=np.asarray(testRecA).T


    c1Pres=presArr[0]
    c2Pres=presArr[1]
    c3Pres=presArr[2]

    c1Rec=recArr[0]
    c2Rec=recArr[1]
    c3Rec=recArr[2]

    yPredict=[]
    yTrue=[]
    for batch,(x,y,fr) in enumerate(testDataLoad):
        validInput=x.to(device)
        realFrameLen=fr


        validationPreds=model(validInput,realFrameLen)
        validIndex=torch.argmax(validationPreds,dim=1)
        yPredict.extend(validIndex.cpu().tolist())
        yTrue.extend(y.cpu().tolist())

   # print("whats here",yTrue)
   # print("whats here 2",yPredict)
    confMatrix=confusion_matrix(yTrue,yPredict,labels=[0,1,2])
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
    plt.savefig('NormVsValLoss.png')
    plt.show()

    plt.plot(c1Pres)
    plt.plot(c2Pres)
    plt.plot(c3Pres)
    plt.title("Model Precision")
    plt.xlabel("Precision")
    plt.ylabel("Epoch")
    plt.legend(['Class 1','Class 2','Class 3'])
    plt.savefig('Precision.png')
    plt.show()

    plt.plot(c1Rec)
    plt.plot(c2Rec)
    plt.plot(c3Rec)
    plt.title("Model Recall")
    plt.xlabel("Recall")
    plt.ylabel("Epoch")
    plt.legend(['Class 1','Class 2','Class 3'])
    plt.savefig('Recall.png')
    plt.show()


def saveModel(model,epoch,optimiser,vLoss,loss):

  modelToSave=model.module if isinstance(model,nn.DataParallel) else model
  torch.save({
      'epoch':epoch,
      'modelStateDict':modelToSave.state_dict(),
      'optimiserStateDict':optimiser.state_dict(),
      'valLoss':vLoss,
      'loss':loss
   },'savedModel.tar')


trainingLoop(50,FullModel,trainLoader,testLoader,lossFunction,validationLossFunction,adamOptimiser,stepLearnDecay,metric,testMetric,testPreicision,testRecall)