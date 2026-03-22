import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.append(r"C:\Users\ASUS\Desktop\Mini Project\Deepfake-video-Detection\ML model\CNN_LSTM")
from model import DeepFakeDetector

print("Imports Successfull !")