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


modelName="google/vit-base-patch16-224"

preTrainedViT=ViTForImageClassification.from_pretrained(modelName)


print(preTrainedViT)