import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')