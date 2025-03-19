import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import tensorboard

#test cuda
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', end='')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))  # Flexes the 5080
    else:
        print(device)

