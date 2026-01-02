import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import torch
from torchvision import io, transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # CNN to map 2D displacement vector into a scalar in range (0,1)

        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2)  
        self.batchnorm1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)  
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)  
        self.batchnorm4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2) 
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm6 = nn.BatchNorm1d(256)

        self.flatten = nn.Flatten(start_dim=1) 

        self.fc1 = nn.Linear(256 * 7, 512)
        self.batchnormflat1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 128)
        self.batchnormflat2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = F.relu(self.batchnorm6(self.conv6(x)))

        x = self.flatten(x)

        x = F.relu(self.batchnormflat1(self.fc1(x)))
        x = F.relu(self.batchnormflat2(self.fc2(x)))

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


    def save(self, phase, save_name="discriminator_model.pth"):
        torch.save(self.state_dict(), f"phase{phase}/weights/"+save_name)



def get_discrim(device, weights_path=''):

    # Getter for Discriminator

    model = Discriminator().to(device)

    if weights_path != '':
        model.load_state_dict(torch.load(weights_path, map_location=device))

    model = nn.DataParallel(model, device_ids=[0,1]).to(device)

    return model
