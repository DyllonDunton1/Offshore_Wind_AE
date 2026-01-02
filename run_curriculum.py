from ae import phase_run
from combine_and_plot_ae import plot_ae
from combine_and_plot_disc import plot_disc
from plot_random import plot_random
from avg_error import get_error


# Main level for running curriculum learning over three phases
# Adjust flags below if you don't want to plot something

phase_start = 1
do_random_plots = True
rand_plot_num = 20
do_error = True

if phase_start == 1: 
    phase_run(1, False, 0, 960)
if phase_start <= 2:
    phase_run(2, False, 448, 512)
if phase_start <= 3:
    phase_run(3, True, 448, 512)

plot_ae()
plot_disc()

if do_random_plots:
    for _ in range(rand_plot_num):
        plot_random()

if do_error:
    out_file = "error_summary.txt"

    with open(out_file, "w") as f:
        f.write("Phase   Split   MAE       Mean      Variance\n")
        f.write("--------------------------------------------\n")

        for phase, split in [
            ("phase1", "train"),
            ("phase1", "test"),
            ("phase2", "train"),
            ("phase2", "test"),
            ("phase3", "train"),
            ("phase3", "test"),
        ]:
            mae, mean, var = get_error(phase, split)
            f.write(f"{phase:<7} {split:<6} {mae:>8.4f} {mean:>9.4f} {var:>9.4f}\n")

    print(f"Saved to {out_file}")

