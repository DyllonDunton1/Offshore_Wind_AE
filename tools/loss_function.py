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


def compute_loss_dc(original_timx, original_timy, reconstructed_timx, reconstructed_timy):
    # Quantify the difference in dc offset between original and reconstruction
    # Define the mean of the signal as proxy for DC offset. Original mean should equal reconstructed mean
    return F.mse_loss(torch.mean(original_timx), torch.mean(reconstructed_timx)) + F.mse_loss(torch.mean(original_timy), torch.mean(reconstructed_timy))

def compute_loss_std(original_timx, original_timy, reconstructed_timx, reconstructed_timy):
    """
    Quantify the difference in oscillation amplitude between original and reconstruction.

    We use the standard deviation of the signal as a proxy for oscillatory amplitude.
    Std removes the DC component automatically and measures how "large" the motion is
    around the equilibrium position.

    Matching std ensures that the reconstructed signal has the correct physical scale,
    even when the absolute signal magnitude is small.
    """
    # Compute per-signal standard deviation over time
    std_orig_x = torch.std(original_timx, dim=-1, unbiased=False)
    std_recon_x = torch.std(reconstructed_timx, dim=-1, unbiased=False)

    std_orig_y = torch.std(original_timy, dim=-1, unbiased=False)
    std_recon_y = torch.std(reconstructed_timy, dim=-1, unbiased=False)

    # Penalize mismatch in oscillation amplitude
    return F.mse_loss(std_orig_x, std_recon_x) + F.mse_loss(std_orig_y, std_recon_y)

def compute_loss_adv(fake_discs, epsilon):
    # This loss function encourages the autoencoder to fool the discriminator into thinking that fake images are real
    # We simply use the discriminator to accomplish this by using a loss function that guides this behavior
    # L_adv = -log(disc(fakes))
    l_adv = -1 * torch.log(torch.mean(fake_discs[:1024]) + epsilon)
    return l_adv


def compute_loss_reg(latent, eps=1e-6):
    # Regularization loss function for guiding latent distribution to N(0,1)

    mu  = latent.mean(dim=0)
    var = latent.var(dim=0, correction=0) + eps

    # KL divergence between N(mu,var) and N(0,1)
    kl = 0.5 * torch.sum(mu**2 + var - torch.log(var) - 1)

    kl_norm = kl / latent.size(1)

    return kl_norm, mu.mean(), var.mean()

def compute_loss_spectral_fft(original_timx, original_timy, reconstructed_timx, reconstructed_timy):
    #FFT Loss function

    eps = 1e-6

    spectral_original_timx = torch.fft.rfft(original_timx, dim=-1)
    spectral_reconstructed_timx = torch.fft.rfft(reconstructed_timx, dim=-1)
    spectral_original_timy = torch.fft.rfft(original_timy, dim=-1)
    spectral_reconstructed_timy = torch.fft.rfft(reconstructed_timy, dim=-1)

    loss_x = F.mse_loss(torch.abs(spectral_original_timx), torch.abs(spectral_reconstructed_timx))
    loss_y = F.mse_loss(torch.abs(spectral_original_timy), torch.abs(spectral_reconstructed_timy))


    return loss_x + loss_y

def compute_loss_spectral_stft(orig_x, orig_y, rec_x, rec_y):
    # STFT Loss function

    fft_sizes = [64, 128, 256] # Different FFT sizes for different time/freq scales
    hop_factors = [4, 4, 4] # 75% coverage

    loss = 0.0
    for n_fft, hop_div in zip(fft_sizes, hop_factors):
        hop = n_fft // hop_div
        win = torch.hann_window(
            n_fft, periodic=True,
            device=orig_x.device,
            dtype=orig_x.dtype
        )

        def stft_mag(sig):
            return torch.stft(
                sig, n_fft=n_fft,
                hop_length=hop,
                window=win,
                center=True,
                return_complex=True
            ).abs()

        Xo = stft_mag(orig_x); Xr = stft_mag(rec_x)
        Yo = stft_mag(orig_y); Yr = stft_mag(rec_y)

        loss += F.mse_loss(Xr, Xo)
        loss += F.mse_loss(Yr, Yo)

    return loss / len(fft_sizes)


def per_sample_std(x, dim=-1, eps=1e-6):
    # Return the standard deviation for each sample in batch for use in time loss
    return torch.sqrt(torch.var(x, dim=dim, unbiased=False) + eps)


def compute_loss_time(original_timx, original_timy, reconstructed_timx, reconstructed_timy, alpha=0.5, w_max=3.0, eps=1e-6):
    # This loss function quantifies the difference in the orignal and reconstructed time series
    # Since the small signals are getting ignored, weight their loss contributions more
    # Detect by small std


    std_orig_x = per_sample_std(original_timx)
    std_orig_y = per_sample_std(original_timy)

    std_ref_x = std_orig_x.mean().detach()
    std_ref_y = std_orig_y.mean().detach()
    w_x = torch.clamp(((std_ref_x / (std_orig_x + eps)) ** alpha), max=w_max)
    w_y = torch.clamp(((std_ref_y / (std_orig_y + eps)) ** alpha), max=w_max)

    mse_per_sample_x = torch.mean((reconstructed_timx - original_timx)**2, dim=-1)
    mse_per_sample_y = torch.mean((reconstructed_timy - original_timy)**2, dim=-1)

    
    l_tim = torch.mean(w_x * mse_per_sample_x) + torch.mean(w_y * mse_per_sample_y)

    return l_tim


def fd_first_derivative(x, dt):
    # Return Central Finited Difference for first derivative
    return (x[:, 2:] - x[:, :-2]) / (2.0 * dt)

def fd_second_derivative(x, dt):
    # Return Central Finited Difference for second derivative
    return (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / (dt ** 2)


def compute_loss_pinn(x, y, wave_data, params, epoch, batch_num, batch_size, device, train_valid):
    # This loss function quantifies how well the reconstruction follows the underlying diff eqs of the system
    # We will use MSE loss between the reconstruction and the equation

    dt = 1/7

    dxdt   = fd_first_derivative(x, dt)
    dydt   = fd_first_derivative(y, dt)

    d2xdt2 = fd_second_derivative(x, dt)
    d2ydt2 = fd_second_derivative(y, dt)

    # Align state to interior points
    x_i = x[:, 1:-1]
    y_i = y[:, 1:-1]
    wave_data_i = wave_data[:, 1:-1]


    
    # -----------------------------------------------------
    # ODE LOSS
    # -----------------------------------------------------
    
    ax_coeffs = torch.tensor([0.000038, 0.027698, -0.002980]).view(1,3).expand(batch_size,3).to(device)
    ay_coeffs = torch.tensor([0.000022, 0.023178, -0.002113]).view(1,3).expand(batch_size,3).to(device)
    bx_coeffs = torch.tensor([0.000001, 0.000720, -0.000081]).view(1,3).expand(batch_size,3).to(device)
    by_coeffs = torch.tensor([0.000001, 0.000744, -0.000091]).view(1,3).expand(batch_size,3).to(device)
    cx_coeffs = torch.tensor([-0.000001, -0.003254, 0.000883]).view(1,3).expand(batch_size,3).to(device)
    cy_coeffs = torch.tensor([0.000001, -0.002900, 0.000832]).view(1,3).expand(batch_size,3).to(device)



    ax = torch.sum(params * ax_coeffs, dim=1).view(-1,1).expand(batch_size,418) - 3.608148
    ay = torch.sum(params * ay_coeffs, dim=1).view(-1,1).expand(batch_size,418) - 3.555526
    bx = torch.sum(params * bx_coeffs, dim=1).view(-1,1).expand(batch_size,418) - 0.004817 
    by = torch.sum(params * by_coeffs, dim=1).view(-1,1).expand(batch_size,418) - 0.004715
    cx = torch.sum(params * cx_coeffs, dim=1).view(-1,1).expand(batch_size,418) + 0.024869
    cy = torch.sum(params * cy_coeffs, dim=1).view(-1,1).expand(batch_size,418) + 0.022017

    theta = params[:,0].view(-1,1).expand(batch_size, 418)
    

    l_pinn_x = d2xdt2 - ax*x_i - bx*dxdt - cx*wave_data_i*torch.cos(theta)
    l_pinn_y = d2ydt2 - ay*y_i - by*dydt - cy*wave_data_i*torch.sin(theta)

    l_pinn = torch.mean(l_pinn_x**2) + torch.mean(l_pinn_y**2)

    
    # -----------------------------------------------------
    # PLOT PINN FOR VIEW
    # -----------------------------------------------------

    if batch_num==0 and epoch%12 == 0:
        t_seconds = np.linspace(0, 60, 421)[:-1]
        t_seconds = t_seconds[1:-1]
        x_np = x_i[0].detach().cpu().numpy()
        dxdt_np = dxdt[0].detach().cpu().numpy()
        d2xdt2_np = d2xdt2[0].detach().cpu().numpy()
        y_np = y_i[0].detach().cpu().numpy()
        dydt_np = dydt[0].detach().cpu().numpy()
        d2ydt2_np = d2ydt2[0].detach().cpu().numpy()

        data = {
            "time": t_seconds,
            "x": x_np,
            "dxdt": dxdt_np,
            "d2xdt2": d2xdt2_np,
            "y": y_np,
            "dydt": dydt_np,
            "d2ydt2": d2ydt2_np
        }
        df = pd.DataFrame(data)

        # Save CSV
        filename = f"phase3/training_plots/pinn_demos/pinn_demo_epoch_{epoch}_{train_valid}.csv"
        df.to_csv(filename, index=False)

        # Plot
        plt.figure(figsize=(12, 6))

        # x and dx/dt
        plt.subplot(2,1,1)
        plt.plot(t_seconds, x_np, label='x')
        plt.plot(t_seconds, dxdt_np, label='dx/dt')
        plt.plot(t_seconds, d2xdt2_np, label='d2x/dt2', linestyle='--')
        plt.title(f"X trajectory and derivatives at epoch {epoch}")
        plt.xlabel("Time [s]")
        plt.ylabel("X / derivatives")
        plt.legend()
        plt.grid(True)

        # y and dy/dt
        plt.subplot(2,1,2)
        plt.plot(t_seconds, y_np, label='y')
        plt.plot(t_seconds, dydt_np, label='dy/dt')
        plt.plot(t_seconds, d2ydt2_np, label='d2y/dt2', linestyle='--')
        plt.title(f"Y trajectory and derivatives at epoch {epoch}")
        plt.xlabel("Time [s]")
        plt.ylabel("Y / derivatives")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"phase3/training_plots/pinn_demos/pinn_demo_epoch_{epoch}_{train_valid}.png", dpi=300, bbox_inches="tight")
        plt.close()



    return l_pinn

import math

def get_weight_function(epoch, w_max=1.0, delay=16, start_epoch=448, ramp_time=40):
    """
    Weight schedule:
    - 0 until a few epochs after the 3rd cosine cycle
    - Smooth cosine ramp
    - Full weight by halfway through the next cycle
    """

    start = start_epoch + delay  
    end = start + ramp_time      

    if epoch < start:
        return 0.0

    if epoch >= end:
        return w_max

    # Cosine ramp from 0 -> w_max
    u = (epoch - start) / (end - start)
    return 0.5 * w_max * (1.0 - math.cos(math.pi * u))


def compute_loss(phase, original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn=False):
    if phase == 1:
        return compute_loss_1(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn)
    elif phase == 2:
        return compute_loss_2(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn)
    elif phase == 3:
        return compute_loss_3(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn)

def compute_loss_1(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn=False):
    tim_weight  = 30.0
    fft_weight  = 0.16
    dc_weight   = 1.0
    std_weight  = 5.0
    reg_weight  = 0.1
    adv_weight  = 0.5
    if epoch_num < 0:
        adv_weight = 0

    recon_time_x = reconstruction[:,0,:]
    recon_time_y = reconstruction[:,1,:]
    origi_time_x = original[:,0,:]
    origi_time_y = original[:,1,:]

    l_tim  = compute_loss_time(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_fft  = compute_loss_spectral_fft(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_dc   = compute_loss_dc(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_std  = compute_loss_std(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_reg, mu, var  = compute_loss_reg(latent)
    l_adv  = compute_loss_adv(fake_discs, epsilon)
    
    # Don't let adv loss dominate
    cap = 0.5 * (l_tim*tim_weight + l_fft*fft_weight) / 5
    l_adv_eff = torch.minimum(l_adv*adv_weight, cap)

    loss = l_tim*tim_weight + l_fft*fft_weight + l_dc*dc_weight + l_std*std_weight + l_reg*reg_weight + l_adv_eff
    
    return loss, l_tim*tim_weight, l_fft*fft_weight, l_dc*dc_weight, l_std*std_weight, l_reg*reg_weight, l_adv_eff, torch.tensor(0), torch.tensor(0), (mu,var)

def compute_loss_2(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn=False):
    tim_weight  = 30.0
    fft_weight  = 0.16
    dc_weight   = 1.0
    std_weight  = 5.0
    reg_weight  = 0.1
    adv_weight  = 0.5
    if epoch_num < 0:
        adv_weight = 0
    stft_weight = get_weight_function(epoch_num, w_max=10*fft_weight)

    recon_time_x = reconstruction[:,0,:]
    recon_time_y = reconstruction[:,1,:]
    origi_time_x = original[:,0,:]
    origi_time_y = original[:,1,:]

    l_tim  = compute_loss_time(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_fft  = compute_loss_spectral_fft(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_dc   = compute_loss_dc(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_std  = compute_loss_std(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_reg, mu, var  = compute_loss_reg(latent)
    l_adv  = compute_loss_adv(fake_discs, epsilon)
    l_stft = compute_loss_spectral_stft(origi_time_x, origi_time_y, recon_time_x, recon_time_y)*stft_weight

    # Don't let adv loss dominate
    spectral_eff = l_fft*fft_weight + l_stft*stft_weight
    cap = 0.5 * (l_tim*tim_weight + spectral_eff) / 5
    l_adv_eff = torch.minimum(l_adv*adv_weight, cap)

    
    loss = l_tim*tim_weight + spectral_eff + l_dc*dc_weight + l_std*std_weight + l_reg*reg_weight + l_adv_eff
    
    return loss, l_tim*tim_weight, l_fft*fft_weight, l_dc*dc_weight, l_std*std_weight, l_reg*reg_weight, l_adv_eff, l_stft*stft_weight, torch.tensor(0), (mu,var)

def compute_loss_3(original, latent, reconstruction, fake_discs, wave_batch, params, epsilon, epoch_num, batch_num, device, train_val, do_pinn=False):
    #print(original.size(), reconstruction.size())
    
    
    tim_weight  = 30.0
    fft_weight  = 0.16
    dc_weight   = 1.0
    std_weight  = 5.0
    reg_weight  = 0.1
    adv_weight  = 0.5
    if epoch_num < 0:
        adv_weight = 0
    pinn_weight = get_weight_function(epoch_num, w_max=1.0)
    stft_weight = fft_weight*10

    recon_time_x = reconstruction[:,0,:]
    recon_time_y = reconstruction[:,1,:]
    origi_time_x = original[:,0,:]
    origi_time_y = original[:,1,:]

    l_tim  = compute_loss_time(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_fft  = compute_loss_spectral_fft(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_dc   = compute_loss_dc(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_std  = compute_loss_std(origi_time_x, origi_time_y, recon_time_x, recon_time_y)
    l_reg, mu, var  = compute_loss_reg(latent)
    l_adv  = compute_loss_adv(fake_discs, epsilon)

    l_pinn = torch.tensor(0)
    if do_pinn:
        batch_size = len(reconstruction)
        l_pinn = compute_loss_pinn(recon_time_x, recon_time_y, wave_batch, params, epoch_num, batch_num, batch_size, device, train_val)
          
    l_stft = compute_loss_spectral_stft(origi_time_x, origi_time_y, recon_time_x, recon_time_y)*stft_weight

    # Don't let adv loss dominate
    spectral_eff = l_fft*fft_weight + l_stft*stft_weight
    cap = 0.5 * (l_tim*tim_weight + spectral_eff) / 5
    l_adv_eff = torch.minimum(l_adv*adv_weight, cap)
    l_pin_eff = torch.minimum(l_pinn*pinn_weight, cap)

    

    loss = l_tim*tim_weight + spectral_eff + l_dc*dc_weight + l_std*std_weight + l_reg*reg_weight + l_adv_eff + l_pin_eff
    
    return loss, l_tim*tim_weight, l_fft*fft_weight, l_dc*dc_weight, l_std*std_weight, l_reg*reg_weight, l_adv_eff, l_stft*stft_weight, l_pin_eff, (mu,var)
