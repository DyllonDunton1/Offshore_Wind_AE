import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tools.tensor_dataset import get_tensor_dataloader
from modules.auto_enc import get_ae


def plot_random():

    valid_path = "./tensors/testing/"
    device = "cuda"
    draw_dataloader, draw_dataset = get_tensor_dataloader(valid_path, 1, device, shuffle=True)

    # Output directory
    out_dir = "random_photos"
    os.makedirs(out_dir, exist_ok=True)

    # Import Models
    auto_enc_weights = "phase3/weights/adv_auto_enc_model.pth"
    model = get_ae(device, weights_path=auto_enc_weights)
    model.eval()

    with torch.no_grad():

        # Grab ONE random batch
        tensor_batch, wave_batch, parameters = next(iter(draw_dataloader))
        tensor_batch = tensor_batch.to(device)

        params = parameters[0]
        print(f"PARAMS: {params}")

        latent, reconstruction = model.module(tensor_batch, t_for_pinn=None)

        orig = tensor_batch[0].detach().cpu()        # (2, 420)
        recon = reconstruction[0].detach().cpu()     # (2, 420)

        # Time vector
        t = np.linspace(0, 60, 421)[:-1]
        dt = t[1] - t[0]
        freqs = np.fft.rfftfreq(len(t), d=dt)

        # Original
        orig_x = orig[0]
        orig_y = orig[1]

        # Reconstruction
        out_x = recon[0]
        out_y = recon[1]

        dir_ = f"{params[0]:.2f}"
        tp   = f"{params[1]:.2f}"
        hs   = f"{params[2]:.2f}"
        # -----------------------
        # FFTs (rFFT)
        # -----------------------
        spectral_original_timx = torch.fft.rfft(orig_x, dim=-1, norm=None)
        spectral_reconstructed_timx = torch.fft.rfft(out_x, dim=-1, norm=None)
        spectral_original_timy = torch.fft.rfft(orig_y, dim=-1, norm=None)
        spectral_reconstructed_timy = torch.fft.rfft(out_y, dim=-1, norm=None)

        spec_orig_x = torch.abs(spectral_original_timx).numpy()
        spec_recon_x = torch.abs(spectral_reconstructed_timx).numpy()
        spec_orig_y = torch.abs(spectral_original_timy).numpy()
        spec_recon_y = torch.abs(spectral_reconstructed_timy).numpy()

        # -----------------------
        # Combined time + FFT subplot figure
        # -----------------------
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

        # X(t)
        axs[0, 0].plot(t, orig_x, label="Original X", linewidth=2)
        axs[0, 0].plot(t, out_x, label="Reconstructed X", linestyle="--", linewidth=2)
        axs[0, 0].set_ylabel("X displacement")
        axs[0, 0].set_title(f"Dir={dir_}, Tp={tp}, Hs={hs}")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Y(t)
        axs[1, 0].plot(t, orig_y, label="Original Y", linewidth=2)
        axs[1, 0].plot(t, out_y, label="Reconstructed Y", linestyle="--", linewidth=2)
        axs[1, 0].set_xlabel("Time (s)")
        axs[1, 0].set_ylabel("Y displacement")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # |FFT(X)|
        axs[0, 1].plot(freqs, spec_orig_x, label="Original X", linewidth=2)
        axs[0, 1].plot(freqs, spec_recon_x, label="Reconstructed X", linestyle="--", linewidth=2)
        axs[0, 1].set_ylabel("|FFT(X)|")
        axs[0, 1].set_title("Spectral Content")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # |FFT(Y)|
        axs[1, 1].plot(freqs, spec_orig_y, label="Original Y", linewidth=2)
        axs[1, 1].plot(freqs, spec_recon_y, label="Reconstructed Y", linestyle="--", linewidth=2)
        axs[1, 1].set_xlabel("Frequency (Hz)")
        axs[1, 1].set_ylabel("|FFT(Y)|")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/xy_time_and_fft_D{dir_}_Tp{tp}_Hs{hs}.png", dpi=150)
        plt.close()
