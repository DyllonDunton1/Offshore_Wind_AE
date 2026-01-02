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
import time




def draw_photos(phase, model, tensor_dataloader, epoch, device, excess_print=False):

    # params: theta, Tp, Hs
    done_10_7_1 = False
    done_40_3_3 = False

    for i, (tensor_batch, wave_batch, parameters) in enumerate(tensor_dataloader):

        if done_10_7_1 and done_40_3_3:
            break

        params = parameters[0]   # (theta, Tp, Hs)

        cond_10_7_1 = (int(params[0]) == 10 and int(params[1]) == 7 and int(params[2]) == 1)
        cond_40_3_3 = (int(params[0]) == 40 and int(params[1]) == 3 and int(params[2]) == 3)

        # Only draw these two cases
        if (not done_10_7_1 and cond_10_7_1) or (not done_40_3_3 and cond_40_3_3):

            tensor_batch = tensor_batch.to(device)

            print(f"PARAMS: {params}")

            latent, reconstruction = model.module(tensor_batch, t_for_pinn=None)



            orig = tensor_batch[0].detach().cpu()  
            recon = reconstruction[0].detach().cpu() 

            # Extract time, x, y
            t = np.linspace(0, 60, 421)[:-1]  
            orig_x = orig[0]
            orig_y = orig[1]
            out_x = recon[0]
            out_y = recon[1]

            # === Plot X(t) ===
            plt.figure(figsize=(10, 4))
            plt.plot(t, orig_x, label="Original X", linewidth=2)
            plt.plot(t, out_x, label="Reconstructed X", linewidth=2, linestyle='--')
            plt.title(f"X Time Series: epoch={epoch}, Hs={int(params[2])}, Tp={int(params[1])}, Dir={int(params[0])}")
            plt.xlabel("Time (s)")
            plt.ylabel("X displacement")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"phase{phase}/photo_test/x_series_{int(params[0])}_{epoch}.png")
            plt.close()

            # === Plot Y(t) ===
            plt.figure(figsize=(10, 4))
            plt.plot(t, orig_y, label="Original Y", linewidth=2)
            plt.plot(t, out_y, label="Reconstructed Y", linewidth=2, linestyle='--')
            plt.title(f"Y Time Series: epoch={epoch}, Hs={int(params[2])}, Tp={int(params[1])}, Dir={int(params[0])}")
            plt.xlabel("Time (s)")
            plt.ylabel("Y displacement")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"phase{phase}/photo_test/y_series_{int(params[0])}_{epoch}.png")
            plt.close()

            # Mark completion
            if cond_40_3_3:
                done_40_3_3 = True
            elif cond_10_7_1:
                done_10_7_1 = True

    
    
    








