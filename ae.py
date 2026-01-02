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


import tools.photo_conversions as pcon
from tools.tensor_dataset import get_tensor_dataloader
from tools.loss_function import compute_loss
from tools.ten2pho import draw_photos
from tools.make_animations import make_combined_gifs
from avg_error import get_error

from modules.auto_enc import get_ae
from modules.discrim import get_discrim


warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def phase_run(phase, do_pinn, start_epoch, num_epochs):
    # Important Training params

    lr = 1e-4
    lr_min = 1e-5

    disc_lr_scaling = 4
    disc_lr = lr_min * disc_lr_scaling

    t_0 = 64
    t_mult = 2
    cosine_cycles = 4

    batch_size = 8440
    disc_batch_size = 1024

    epsilon = 1e-8

    disc_print_freq = 20
    print_freq = 2
    print_mem = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate DataLoaders for training and validation
    tensor_path = "./tensors/training/"
    valid_path = "./tensors/testing/"
    disc_dataloader, disc_dataset = get_tensor_dataloader(tensor_path, disc_batch_size, device, shuffle=True)
    tensor_dataloader, tensor_dataset = get_tensor_dataloader(tensor_path, batch_size, device, shuffle=True)
    valid_dataloader, valid_dataset = get_tensor_dataloader(valid_path, batch_size, device, shuffle=False)
    draw_dataloader, draw_dataset = get_tensor_dataloader(valid_path, 1, device, shuffle=False)
    print(len(tensor_dataset), len(valid_dataset))


    # Import Models
    if phase == 1:
        auto_enc_weights = ''
    else:
        auto_enc_weights = f'phase{phase-1}/weights/adv_auto_enc_model.pth'
    discrim_weights = ''

    model = get_ae(device, weights_path=auto_enc_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_mult, eta_min=lr_min)

    discriminator = get_discrim(device, weights_path=discrim_weights)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=disc_lr)

    # Gather Start time
    now_time = time.time()
    min_epoch_loss = np.inf



    # Set up ae_log and disc_log
    ae_log_columns = [
        "Epoch", "Learning Rate",
        "Training Loss", "Time Loss", "FFT Loss", "Offset Loss", "Amplitude Loss", "Regularization Loss", "Adversarial Loss", "STFT Loss", "PINN Loss", "Error", "Mean", "Var",
        "Testing Loss", "Testing Time Loss", "Testing FFT Loss", "Testing Offset Loss", "Testing Amplitude Loss", "Testing Regularization Loss", "Testing Adversarial Loss", "Testing STFT Loss", "Testing PINN Loss","Testing Error", "Testing Mean", "Testing Var"
    ]
    ae_logs = pd.DataFrame(columns=ae_log_columns)
    disc_logs = pd.DataFrame(columns=["Epoch", "Disc LR", "D per G", "Disc Loss", "Real Guess", "Real Quant", "Fake Guess", "Fake Quant"])
    lr_inspection_logs = pd.DataFrame(columns=["Epoch", "D LR", "G LR"])

    for epoch in range(start_epoch, start_epoch + num_epochs):

        print(f"Epoch {epoch}")

        if epoch < 10:
            disc_high_cutoff = 0.70
            disc_low_cutoff = 0.55
            max_d_per_g = 2
        elif epoch < 180:
            disc_high_cutoff = 0.75
            disc_low_cutoff = 0.50
            max_d_per_g = 1
        else:
            disc_high_cutoff = 0.78
            disc_low_cutoff = 0.48
            max_d_per_g = 1

        print("Training Discriminator")

        discriminator.train()
        model.eval()

        for p in discriminator.parameters():
            p.requires_grad = True

        disc_epoch = 0
        running_disc_fake = 0.0
        running_disc_real = 0.0
        running_disc_loss = 0.0
        running_disc_fake_quant = 0.0
        running_disc_real_quant = 0.0
        disc_iterations = 0

        # -----------------------------
        # Train discriminator (up to max_d_per_g epochs)
        # -----------------------------
        for d_epoch in range(max_d_per_g):

            for i, (tensor_batch, wave_batch, params) in enumerate(disc_dataloader):

                if i % disc_print_freq == 0:
                    print(f"{i * disc_batch_size}/{len(tensor_dataset)}")
                    if print_mem:
                        print(f"Memory Allocated after pull in data: "
                            f"{torch.cuda.memory_allocated(device)/1e9} GB")

                tensor_batch = tensor_batch.to(device)
                disc_optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    _, reconstructions = model(tensor_batch)

                # Discriminator forward
                discriminations_real = discriminator(tensor_batch)
                discriminations_fake = discriminator(reconstructions)

                fake_mean = discriminations_fake.mean().item()
                real_mean = discriminations_real.mean().item()

                l_disc_real = -1*torch.log(torch.mean(discriminations_real)) # Should force these real discs to 1
                l_disc_fake = -1*torch.log(1 - torch.mean(discriminations_fake) + 1e-8) # Should force these fake discs to 0
                l_disc = l_disc_real + l_disc_fake

                running_disc_loss += l_disc.item()
                running_disc_fake += fake_mean
                running_disc_real += real_mean
                disc_iterations += 1

                # Percentile-based early stop check 
                real_p10 = torch.quantile(discriminations_real.detach(), 0.10).item()
                fake_p90 = torch.quantile(discriminations_fake.detach(), 0.90).item()
                running_disc_fake_quant += fake_p90
                running_disc_real_quant += real_p10

                if real_p10 >= disc_high_cutoff and fake_p90 <= disc_low_cutoff:
                    print(f"Skipping Disc Training since "
                        f"Real={real_p10:.3f} >= {disc_high_cutoff} and "
                        f"Fake={fake_p90:.3f} <= {disc_low_cutoff}")
                    break

                l_disc.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()

            disc_epoch += 1

            # Epoch-level mean check (unchanged logic)
            mean_real = running_disc_real / max(1, disc_iterations)
            mean_fake = running_disc_fake / max(1, disc_iterations)

            print(f"REAL GUESS: {mean_real:.4f}  | FAKE GUESS: {mean_fake:.4f}")

            

        disc_logs = pd.concat([
            disc_logs,
            pd.DataFrame([{
                "Epoch": epoch,
                "Disc LR": disc_optimizer.param_groups[0]['lr'],
                "D per G": disc_epoch,
                "Disc Loss": running_disc_loss/disc_iterations,
                "Real Guess": running_disc_real/disc_iterations,
                "Real Quant": real_p10,
                "Fake Guess": running_disc_fake/disc_iterations,
                "Fake Quant": fake_p90
            }])
        ], ignore_index=True)

        disc_logs.to_csv(f"phase{phase}/training_plots/disc/disc_data.csv", index=False)
        
        disc_logs.plot(x="Epoch", y=["Disc Loss"])
        plt.title(f"Disc Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/disc/disc_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        disc_logs.plot(x="Epoch", y=["Real Guess", "Fake Guess"])
        plt.title(f"Disc Guesses vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Guess")
        plt.legend(["Real", "Fake"])
        plt.axhline(y=disc_high_cutoff, color='black', linestyle='--')
        plt.axhline(y=disc_low_cutoff, color='black', linestyle='--')
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/disc/disc_guesses.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        disc_logs.plot(x="Epoch", y=["Real Quant", "Fake Quant"])
        plt.title(f"Disc Guess Quantiles vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Guess")
        plt.legend(["Real", "Fake"])
        plt.axhline(y=disc_high_cutoff, color='black', linestyle='--')
        plt.axhline(y=disc_low_cutoff, color='black', linestyle='--')
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/disc/disc_quants.png", dpi=300, bbox_inches="tight")
        plt.close()

        disc_logs.plot(x="Epoch", y=["Disc LR"])
        plt.title(f"Disc LR vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Disc LR")
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/disc/disc_lr.png", dpi=300, bbox_inches="tight")
        plt.close()

        disc_logs.plot(x="Epoch", y=["D per G"])
        plt.title(f"D epochs per G vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("D per G")
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/disc/d_per_g.png", dpi=300, bbox_inches="tight")
        plt.close() 
        

        print(disc_logs.tail(1).to_string(index=False))

        discriminator.module.save(phase)
        model.module.save(phase, extra=f"_epoch_{epoch}")

        time_elapsed = time.time() - now_time
        print(time_elapsed)
        hours_elapsed = time_elapsed/3600

        print(f"Time Elapsed: {time_elapsed/3600.0} Hours")


        
        print("Training AutoEncoder")
        for p in discriminator.parameters():
            p.requires_grad = False
        model.train()
        running_encoder_loss = 0
        running_loss_tim = 0
        running_loss_fft = 0
        running_loss_dc  = 0
        running_loss_std = 0
        running_loss_reg = 0
        running_loss_adv = 0
        running_loss_stft = 0
        running_loss_pinn = 0
        running_mu = 0
        running_var = 0
        encoder_iterations = 0

        # -----------------------------
        # Train AutoEncoder
        # -----------------------------

        for i, (tensor_batch, wave_batch, params) in enumerate(tensor_dataloader):
            if i%print_freq == 0:
                print(f"{i*batch_size}/{len(tensor_dataset)}")
                if print_mem:
                    print(f"Memory Allocated after pull in data: {torch.cuda.memory_allocated(device)/1e9} GB")

            tensor_batch = tensor_batch.to(device)
            wave_batch = wave_batch.to(device)
            params = params.to(device)
            optimizer.zero_grad()
            
            latents, reconstructions = model.module(tensor_batch)
            fake_discriminations = discriminator(reconstructions)
            fake_discriminations = torch.clamp(fake_discriminations, min=1e-8, max=1-1e-8)

            loss, loss_tim, loss_fft, loss_dc, loss_std, loss_reg, loss_adv, loss_stft, loss_pinn, (mu, var) = compute_loss(phase, tensor_batch, latents, reconstructions, fake_discriminations, wave_batch, params, epsilon, epoch, i, device, "train", do_pinn=do_pinn)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_encoder_loss += loss.item()
            running_loss_tim += loss_tim.item()
            running_loss_fft += loss_fft.item()
            running_loss_dc  += loss_dc.item()
            running_loss_std += loss_std.item()
            running_loss_reg += loss_reg.item()
            running_loss_adv += loss_adv.item()
            running_loss_stft += loss_stft.item()
            running_loss_pinn += loss_pinn.item()
            running_mu += mu.item()
            running_var += var.item()
            encoder_iterations += 1    

        lr_inspection_logs = pd.concat([
            lr_inspection_logs,
            pd.DataFrame([{
                "Epoch": epoch,
                "D LR": disc_optimizer.param_groups[0]['lr'],
                "G LR": optimizer.param_groups[0]['lr'],
            }])
        ], ignore_index=True)

        lr_inspection_logs.to_csv(f"phase{phase}/training_plots/lr/lr_log.csv", index=False)
        
        lr_inspection_logs.plot(x="Epoch", y=["D LR", "G LR"])
        plt.title(f"D and G LR vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/lr/lr_logs.png", dpi=300, bbox_inches="tight")
        plt.close()

        scheduler.step(epoch)
        time_elapsed = time.time() - now_time
        print(time_elapsed)
        hours_elapsed = time_elapsed/3600

        print(f"Time Elapsed: {time_elapsed/3600.0} Hours")


        print("Testing") 
        model.eval()
        running_val_loss = 0
        running_val_loss_tim = 0
        running_val_loss_fft = 0
        running_val_loss_dc  = 0
        running_val_loss_std = 0
        running_val_loss_reg = 0
        running_val_loss_adv = 0
        running_val_loss_stft = 0
        running_val_loss_pinn = 0
        running_val_mu = 0
        running_val_var = 0
        val_iterations = 0

        # -----------------------------
        # Test AutoEncoder
        # -----------------------------

        for i, (vaild_batch, wave_batch, params) in enumerate(valid_dataloader):
            if i%print_freq == 0:
                print(f"{i*batch_size}/{len(valid_dataset)}")
                if print_mem:
                    print(f"Memory Allocated after pull in data: {torch.cuda.memory_allocated(device)/1e9} GB")

            vaild_batch = vaild_batch.to(device)
            wave_batch = wave_batch.to(device)
            params = params.to(device)
            
            
            latents, reconstructions = model.module(vaild_batch)
            fake_discriminations = discriminator(reconstructions)
            fake_discriminations = torch.clamp(fake_discriminations, min=1e-8, max=1-1e-8)

            valid_loss, loss_tim, loss_fft, loss_dc, loss_std, loss_reg, loss_adv, loss_stft, loss_pinn, (mu, var) = compute_loss(phase, vaild_batch, latents, reconstructions, fake_discriminations, wave_batch, params, epsilon, epoch, i, device, "valid", do_pinn=do_pinn)

            running_val_loss += valid_loss.item()
            running_val_loss_tim += loss_tim.item()
            running_val_loss_fft += loss_fft.item()
            running_val_loss_dc  += loss_dc.item()
            running_val_loss_std += loss_std.item()
            running_val_loss_reg += loss_reg.item()
            running_val_loss_adv += loss_adv.item()
            running_val_loss_stft += loss_stft.item()
            running_val_loss_pinn += loss_pinn.item()
            running_val_mu += mu.item()
            running_val_var += var.item()
            val_iterations += 1
        

        error = 0 #Not currently used. Error calculation mid run not working well. Will fix later
        val_error = 0 # ^^^^^^^^^^^^^^^^^^6

        epoch_log = {
            "Epoch": epoch,
            "Learning Rate": optimizer.param_groups[0]['lr'],
            "Training Loss": running_encoder_loss/encoder_iterations,
            "Time Loss": running_loss_tim/encoder_iterations,
            "FFT Loss": running_loss_fft/encoder_iterations,
            "Offset Loss": running_loss_dc/encoder_iterations,
            "Amplitude Loss": running_loss_std/encoder_iterations,
            "Regularization Loss": running_loss_reg/encoder_iterations,
            "Adversarial Loss": running_loss_adv/encoder_iterations,
            "STFT Loss": running_loss_stft/encoder_iterations,
            "PINN Loss": running_loss_pinn/encoder_iterations,
            "Error": 1000*error,
            "Mean": running_mu/encoder_iterations,
            "Var": running_var/encoder_iterations,
            "Testing Loss": running_val_loss/val_iterations,
            "Testing Time Loss": running_val_loss_tim/val_iterations,
            "Testing FFT Loss": running_val_loss_fft/val_iterations,
            "Testing Offset Loss": running_val_loss_dc/val_iterations,
            "Testing Amplitude Loss": running_val_loss_std/val_iterations,
            "Testing Regularization Loss": running_val_loss_reg/val_iterations,
            "Testing Adversarial Loss": running_val_loss_adv/val_iterations,
            "Testing STFT Loss": running_val_loss_stft/val_iterations,
            "Testing PINN Loss": running_val_loss_pinn/val_iterations,
            "Testing Error": 1000*val_error,
            "Testing Mean": running_val_mu/val_iterations,
            "Testing Var": running_val_var/val_iterations
        }
        ae_logs = pd.concat([ae_logs, pd.DataFrame([epoch_log])], ignore_index=True)
        ae_logs.to_csv(f"phase{phase}/training_plots/ae/ae_logs.csv", index=False)

        plot_vals = ["Training Loss", "Time Loss", "FFT Loss", "Offset Loss", "Amplitude Loss", "Regularization Loss", "Adversarial Loss", "STFT Loss", "PINN Loss"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0,0.5))
        plt.title(f"Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Training_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Testing Loss", "Testing Time Loss", "Testing FFT Loss", "Testing Offset Loss", "Testing Amplitude Loss", "Testing Regularization Loss", "Testing Adversarial Loss", "Testing STFT Loss", "Testing PINN Loss"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0,0.5))
        plt.title(f"Testing Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Validation_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Learning Rate"]
        ae_logs.plot(x="Epoch", y=plot_vals)
        plt.title(f"Learning Rate vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Learning_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Training Loss", "Testing Loss"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0,0.5))
        plt.title(f"Traing and Validation vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Train_vs_Valid_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Mean", "Testing Mean"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(-0.5,0.5))
        plt.title(f"Traing and Validation Latent Mean vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Mean")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Mean_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Var", "Testing Var"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0,6))
        plt.title(f"Traing and Validation Latent Variance vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Variance")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Var_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        plot_vals = ["Error", "Testing Error"]
        ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0,100))
        plt.title(f"Traing and Validation Error vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Error (mm)")
        plt.legend(plot_vals)
        plt.grid(True)
        plt.savefig(f"phase{phase}/training_plots/ae/Error_Plot.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(ae_logs.tail(1).to_string(index=False))

        model.module.save(phase)

        time_elapsed = time.time() - now_time
        print(time_elapsed)
        hours_elapsed = time_elapsed/3600

        print(f"Time Elapsed: {time_elapsed/3600.0} Hours")

        draw_photos(phase, model, draw_dataloader, epoch, device)
        if epoch%64==0 or epoch==959 or epoch in [10,16,32,48]:
            make_combined_gifs(phase, total_epochs=(epoch+1))



    time_elapsed = time.time() - now_time
    print(time_elapsed)
    hours_elapsed = time_elapsed/3600

    print(f"Time Elapsed: {time_elapsed/3600.0} Hours")
