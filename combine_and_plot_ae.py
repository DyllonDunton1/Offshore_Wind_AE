import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_ae():
    # -----------------------
    # Input CSV paths (3 phases)
    # -----------------------
    csv_phase_1 = "phase1/training_plots/ae/ae_logs.csv"
    csv_phase_2 = "phase2/training_plots/ae/ae_logs.csv"
    csv_phase_3 = "phase3/training_plots/ae/ae_logs.csv"

    # -----------------------
    # Epoch offsets
    # -----------------------
    phase1_offset = 0
    phase2_offset = 960 - 448
    phase3_offset = 960 + 512 - 448

    # -----------------------
    # Output folder + combined CSV path
    # -----------------------
    out_dir = "output_plots/"
    os.makedirs(out_dir, exist_ok=True)

    combined_csv = os.path.join(out_dir, "combined_curriculum.csv")

    # -----------------------
    # Load + offset epochs + tag phases
    # -----------------------
    df1 = pd.read_csv(csv_phase_1)
    df1["Epoch"] = df1["Epoch"] + phase1_offset
    df1["Phase"] = 1

    df2 = pd.read_csv(csv_phase_2)
    df2["Epoch"] = df2["Epoch"] + phase2_offset
    df2["Phase"] = 2

    df3 = pd.read_csv(csv_phase_3)
    df3["Epoch"] = df3["Epoch"] + phase3_offset
    df3["Phase"] = 3

    # -----------------------
    # Combine
    # -----------------------
    ae_logs = pd.concat([df1, df2, df3], ignore_index=True)
    ae_logs.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV -> {combined_csv}  (rows={len(ae_logs)})")

    # -----------------------
    # Phase boundaries (global epoch space)
    # -----------------------
    b1 = df1["Epoch"].iloc[-1]
    b2 = df2["Epoch"].iloc[-1]

    def add_phase_lines(ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axvline(b1, linestyle="--", linewidth=1, color="black")
        ax.axvline(b2, linestyle="--", linewidth=1, color="black")

    # -----------------------
    # Plots
    # -----------------------

    # Training loss components
    plot_vals = [
        "Training Loss",
        "Time Loss",
        "FFT Loss",
        "STFT Loss",
        "Offset Loss",
        "Amplitude Loss",
        "Regularization Loss",
        "Adversarial Loss",
        "PINN Loss",
    ]
    ax = ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0, 0.03))
    add_phase_lines(ax)
    ax.set_title("Training Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(plot_vals)
    ax.grid(True)
    plt.savefig(os.path.join(out_dir, "Training_Plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Testing loss components
    plot_vals = [
        "Testing Loss",
        "Testing Time Loss",
        "Testing FFT Loss",
        "Testing STFT Loss",
        "Testing Offset Loss",
        "Testing Amplitude Loss",
        "Testing Regularization Loss",
        "Testing Adversarial Loss",
        "Testing PINN Loss",
    ]
    ax = ae_logs.plot(x="Epoch", y=plot_vals, ylim=(0, 0.03))
    add_phase_lines(ax)
    ax.set_title("Testing Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(plot_vals)
    ax.grid(True)
    plt.savefig(os.path.join(out_dir, "Validation_Plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Learning rate
    plot_vals = ["Learning Rate"]
    ax = ae_logs.plot(x="Epoch", y=plot_vals)
    add_phase_lines(ax)
    ax.set_title("Learning Rate vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True)
    plt.savefig(os.path.join(out_dir, "Learning_Plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Train vs test loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        ae_logs["Epoch"],
        ae_logs["Training Loss"],
        label="Training Loss",
        color="tab:blue",
        linestyle="-"
    )

    plt.plot(
        ae_logs["Epoch"],
        ae_logs["Testing Loss"],
        label="Testing Loss",
        color="tab:orange",
        linestyle="--"
    )

    add_phase_lines()
    plt.title("Training and Validation vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 0.1)
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "Train_vs_Valid_Plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------
    # Latent Mean + Variance (subplot)
    # -----------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---- Mean (top) ----
    axs[0].plot(
        ae_logs["Epoch"],
        ae_logs["Mean"],
        label="Training Mean",
        color="tab:blue",
        linestyle="-",
        linewidth=2
    )
    axs[0].plot(
        ae_logs["Epoch"],
        ae_logs["Testing Mean"],
        label="Testing Mean",
        color="tab:orange",
        linestyle=(0, (5, 5)),
        linewidth=2
    )

    add_phase_lines(axs[0])
    axs[0].set_title("Training and Testing Latent Mean vs Epoch")
    axs[0].set_ylabel("Mean")
    axs[0].set_ylim(-0.05, 0.05)
    axs[0].legend()
    axs[0].grid(True)

    # ---- Variance (bottom) ----
    axs[1].plot(
        ae_logs["Epoch"],
        ae_logs["Var"],
        label="Training Variance",
        color="tab:blue",
        linestyle="-",
        linewidth=2
    )
    axs[1].plot(
        ae_logs["Epoch"],
        ae_logs["Testing Var"],
        label="Testing Variance",
        color="tab:orange",
        linestyle=(0, (5, 5)),
        linewidth=2
    )

    add_phase_lines(axs[1])
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Variance")
    axs[1].set_ylim(0.5, 1.5)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Latent_Mean_Var_Plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

