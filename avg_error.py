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
import PIL
from torchvision.transforms.functional import to_pil_image

from tools.tensor_dataset import get_tensor_dataloader

from modules.auto_enc import get_ae
from modules.discrim import get_discrim

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

batch_size = 8440

valid_path = "./tensors/testing/"
valid_dataloader, valid_dataset = get_tensor_dataloader(valid_path, batch_size, shuffle=True)

tensor_path = "./tensors/training/"
tensor_dataloader, tensor_dataset = get_tensor_dataloader(tensor_path, batch_size, shuffle=True)


auto_enc_weights = f'weights/adv_auto_enc_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_error(phase, train_type, model=None):

    loader = valid_dataloader
    if train_type == "train":
        loader = tensor_dataloader

    if model == None:
        model = get_ae(device, weights_path=f"{phase}/{auto_enc_weights}", parallel=True)

    model.eval()

    sum_of_mag_error = 0
    sum_of_means = 0
    sum_of_vars = 0
    batches_ran = 0

    with torch.no_grad():
        for i, (tensor_batch, wave_batch, params) in enumerate(loader):

            original = tensor_batch.to(device)
            latent, reconstruction = model.module(original)

            print(original.shape, latent.shape)

            recon_time_x = reconstruction[:,0,:]
            recon_time_y = reconstruction[:,1,:]
            origi_time_x = original[:,0,:]
            origi_time_y = original[:,1,:]

            original_mag = (origi_time_x**2 + origi_time_y**2)**0.5
            reconstructed_mag = (recon_time_x**2 + recon_time_y**2)**0.5

            mag_error = F.l1_loss(original_mag, reconstructed_mag)


            means  = latent.mean(dim=0)
            vars = latent.var(dim=0, correction=0)
            
            
            mean = means.mean()
            var = vars.mean()


            sum_of_mag_error += mag_error.item()
            sum_of_means += mean.item()
            sum_of_vars += var.item()
            batches_ran += 1
            

    mag_error_complete = sum_of_mag_error / batches_ran
    print(mag_error_complete)


    avg_mean_latent = sum_of_means / batches_ran
    avg_variance_latent = sum_of_vars / batches_ran
    print(f"MEAN: {avg_mean_latent}")
    print(f"VAR: {avg_variance_latent}")

    model.train()

    return mag_error_complete, avg_mean_latent, avg_variance_latent
