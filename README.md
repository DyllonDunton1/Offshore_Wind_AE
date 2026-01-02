# Offshore_Wind_AE
Official public repository for "A Physics-Informed Latent Representation for Digital Twin Modeling of Tower-Top Motions During Offshore Wind Turbine Installation"


All AE code is written by Dyllon Dunton

Dataset: 

To run code, download dataset file "tensors.zip" and extract to the main level. Then run in terminal:

python3 run_curriculum.py

Training Plots will appear dynamically in '/Phase1', '/Phase2', and '/Phase3'. After CL is complete, combined output plots will be in '/output_plots' and a number of random signals will be pulled from the dataset and reconstructed for visual assessment in '/random_plots'. A final summary of error and latent regularization will be in '/error_summary.txt'.
