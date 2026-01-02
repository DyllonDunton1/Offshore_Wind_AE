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

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


class TensorDataset(Dataset):
    def __init__(self, path, period=None, color_jitter=True):

        # Dataloader for the 2d motion vectors

        self.X = np.load(os.path.join(path, "X.npy"), mmap_mode="r")  # shape (N,2,420)
        self.W = np.load(os.path.join(path, "W.npy"), mmap_mode="r")  # shape (N,420)
        self.P = np.load(os.path.join(path, "P.npy"), mmap_mode="r")  # shape (N,4)   
        self.dtype=torch.float32
   
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Load tensor
        pos_data   = torch.from_numpy(self.X[idx])         # (2,420)
        wave_data  = torch.from_numpy(self.W[idx])         # (420,)
        params     = torch.from_numpy(self.P[idx][0:3])    # (3,)

        return pos_data, wave_data, params



def get_tensor_dataloader(directory, batch_size, period=None, shuffle=False):
    tensor_dataset = TensorDataset(directory, period=period, color_jitter=False)
    tensor_dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
    return tensor_dataloader, tensor_dataset