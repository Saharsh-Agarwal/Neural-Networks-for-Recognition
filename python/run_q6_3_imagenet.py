import numpy as np
import scipy.io
import os
import sys
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

