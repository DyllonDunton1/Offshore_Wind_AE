import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_disc():

    # -----------------------
    # Input CSV paths (3 phases)
    # -----------------------
    csv_phase_1 = "phase1/training_plots/disc/disc_data.csv"
    csv_phase_2 = "phase2/training_plots/disc/disc_data.csv"
    csv_phase_3 = "phase3/training_plots/disc/disc_data.csv"

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

    combined_csv = os.path.join(out_dir, "combined_disc_curriculum.csv")

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
    disc_logs = pd.concat([df1, df2, df3], ignore_index=True)
    disc_logs.to_csv(combined_csv, index=False)
    print(f"Saved combined CSV -> {combined_csv}  (rows={len(disc_logs)})")

    # -----------------------
    # Phase boundaries (global epoch space)
    # -----------------------
    b1 = df1["Epoch"].iloc[-1]
    b2 = df2["Epoch"].iloc[-1]

    def add_phase_lines():
        plt.axvline(b1, linestyle="--", linewidth=1, color="black")
        plt.axvline(b2, linestyle="--", linewidth=1, color="black")

    # -----------------------
    # Dynamic discriminator thresholds (piecewise by epoch)
    # + vertical connectors where they jump
    # -----------------------
    epochs = disc_logs["Epoch"].values
    e_min, e_max = epochs.min(), epochs.max()

    regions = [
        (e_min, 10,    0.70, 0.55),
        (10,   180,    0.75, 0.50),
        (180,  e_max,  0.78, 0.48),
    ]

    def add_threshold_lines():
        # Horizontal segments
        for e_start, e_end, high, low in regions:
            plt.hlines(high, e_start, e_end, colors="black", linestyles="--", linewidth=1)
            plt.hlines(low,  e_start, e_end, colors="black", linestyles="--", linewidth=1)

        # Vertical connectors at boundaries (between consecutive regions)
        for k in range(len(regions) - 1):
            _, e_boundary, high_a, low_a = regions[k]
            _, _,          high_b, low_b = regions[k + 1]
            plt.vlines(e_boundary, high_a, high_b, colors="black", linestyles="--", linewidth=1)
            plt.vlines(e_boundary, low_a,  low_b,  colors="black", linestyles="--", linewidth=1)

    # -----------------------
    # Plot 1: Disc guesses 
    # -----------------------
    disc_logs.plot(x="Epoch", y=["Real Guess", "Fake Guess"])
    add_phase_lines()
    add_threshold_lines()
    plt.title("Disc Guesses vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Guess")
    plt.legend(["Real", "Fake"])
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "disc_guesses.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------
    # Plot 2: Disc quantiles 
    # -----------------------
    disc_logs.plot(x="Epoch", y=["Real Quant", "Fake Quant"])
    add_phase_lines()
    add_threshold_lines()
    plt.title("Disc Quantiles vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Quantile")
    plt.legend(["Real Quant", "Fake Quant"])
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "disc_quantiles.png"), dpi=300, bbox_inches="tight")
    plt.close()
