import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
import torch
from torchvision import io, transforms, models
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import gc
from torch.cuda.amp import autocast, GradScaler
import time
import csv
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


class DeterministicEncoder(nn.Module):
    def __init__(self, input_channels=2, latent_dim=256):
        super(DeterministicEncoder, self).__init__()

        # Encode 2D displacement vector into latent

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.batchnorm1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)         
        self.batchnorm2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)           
        self.batchnorm3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)       
        self.batchnorm4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnorm5 = nn.BatchNorm1d(256)

        self.fc_latent = nn.Linear(256 * 27, latent_dim)

    def forward(self, x):

        x = F.gelu(self.batchnorm1(self.conv1(x)))
        x = F.gelu(self.batchnorm2(self.conv2(x)))
        x = F.gelu(self.batchnorm3(self.conv3(x)))
        x = F.gelu(self.batchnorm4(self.conv4(x)))
        x = F.gelu(self.batchnorm5(self.conv5(x)))

        # flatten
        x = x.view(x.size(0), -1)
        latent = self.fc_latent(x)

        return latent



class DeterministicDecoder(nn.Module):
    def __init__(self, output_channels=2, latent_dim=256):
        super(DeterministicDecoder, self).__init__()
        
        # Decode the latent displacement representation back into 2D vector

        self.fc = nn.Linear(latent_dim, 256 * 27)

        self.convT1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=0)  
        self.batchnorm1 = nn.BatchNorm1d(128)

        self.convT2 = nn.ConvTranspose1d(128, 64,  kernel_size=5, stride=2, padding=2, output_padding=0) 
        self.batchnorm2 = nn.BatchNorm1d(64)

        self.convT3 = nn.ConvTranspose1d(64, 32,   kernel_size=5, stride=2, padding=2, output_padding=1) 
        self.batchnorm3 = nn.BatchNorm1d(32)

        self.convT4 = nn.ConvTranspose1d(32, 2,    kernel_size=5, stride=2, padding=2, output_padding=1) 

        self.tanh = nn.Tanh()

    def forward(self, z):

        x = self.fc(z)
        x = x.view(x.size(0), 256, 27)

        x = F.relu(self.convT1(x))
        x = F.relu(self.convT2(x))
        x = F.relu(self.convT3(x))

        x = self.convT4(x)

        decoded = x
        return decoded






class DeterministicAutoEncoder(nn.Module):
    def __init__(self, freeze_enc=False, original_channels=2, latent_dim=256):
        super(DeterministicAutoEncoder, self).__init__()

        # AutoEncoder assembly encodes 2D displacement vector into latent representation and back again.

        self.encoder = DeterministicEncoder(input_channels=original_channels,
                                            latent_dim=latent_dim)

        self.decoder = DeterministicDecoder(output_channels=2,
                                            latent_dim=latent_dim)


    def forward(self, x, t_for_pinn=None):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)

        return latent, reconstruction

    def save(self, phase, autoencoder_save_name="adv_auto_enc_model", decoder_save_name="adv_dec_model", extra=""):
        print(f"Checkpointing Models")
        torch.save(self.state_dict(), f"phase{phase}/weights/"+autoencoder_save_name+extra+".pth")




def get_ae(device, weights_path='', parallel=True, freeze_enc=False):

    # Getter for the AE

    model = DeterministicAutoEncoder(freeze_enc=freeze_enc).to(device)

    if weights_path != '':
        model.load_state_dict(torch.load(weights_path, map_location=device))

    if parallel:
        model = nn.DataParallel(model, device_ids=[0,1]).to(device)

    return model