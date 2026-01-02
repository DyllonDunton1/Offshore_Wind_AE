# Offshore_Wind_AE

Official public repository for the paper:

**“A Physics-Informed Autoencoder Framework for Digital Twin Modeling of Tower-Top Motions During Offshore Wind Turbine Installation”**

This repository contains the full implementation of the **physics-informed autoencoder (AE)** used to learn a compact latent representation of offshore wind turbine tower-top motion during blade-mating operations. The trained AE is designed to be used as the latent encoder/decoder component of a **latent diffusion digital twin**, enabling real-time, physics-consistent motion prediction.

<img width="2161" height="670" alt="image" src="https://github.com/user-attachments/assets/09aadd25-b11a-406b-925c-ae292ce805f2" />


---

## Authors

- **Dyllon Dunton**  
  Department of Electrical and Computer Engineering, University of Maine  

- **Saravanan Bhaskaran**  
  Department of Mechanical Engineering, University of Maine  

- **Yifeng Zhu**  
  Department of Electrical and Computer Engineering, University of Maine  

- **Andrew Goupee**  
  Department of Mechanical Engineering, University of Maine  

- **Amrit Verma**  
  Department of Mechanical Engineering, University of Maine  

---

## Author Contributions

- **Dyllon Dunton**
  - Designed and implemented the physics-informed autoencoder architecture
  - Implemented curriculum learning strategy and training schedule
  - Implemented adversarial (GAN-style) training
  - Implemented latent-space regularization for diffusion compatibility
  - Developed all AE training, evaluation, and visualization code
  - Performed reconstruction analysis and error evaluation

- **Saravanan Bhaskaran**
  - Designed and executed the OrcaFlex simulation campaign
  - Generated the full high-fidelity dataset of tower-top motions
  - Developed reduced-order governing equations using SINDy
  - Provided physical interpretation of installation-stage dynamics

- **Yifeng Zhu, Andrew Goupee, Amrit Verma**
  - Provided technical guidance on offshore wind turbine dynamics and model formulation
  - Reviewed and refined methodology and results

---

## Repository Scope

This repository contains **only the autoencoder component** of the full digital twin framework.  
The latent diffusion model (DDIM/LDM) is described in the paper but is **not included here**.

Specifically, this repository provides:
- Physics-informed autoencoder (AE)
- Curriculum learning across three training phases
- Time-domain, frequency-domain (FFT & STFT), adversarial, and physics-based losses
- Latent regularization compatible with diffusion models
- Training and validation visualization tools

---

## How This Maps to the Paper

| Paper Section | Repository Component |
|---------------|---------------------|
| Dataset Generation | Pre-generated tensors from OrcaFlex simulations |
| Physics-Informed AE | `run_curriculum.py`, model and loss definitions |
| Curriculum Learning | `ae.py` and `tools/loss_function.py` |
| Reconstruction Results | Auto-generated plots in `/random_plots` |
| Training Curves | Auto-generated plots in `/Phase1`, `/Phase2`, `/Phase3` |
| Latent Statistics | Mean/variance tracking and summary outputs |

---

## Dataset

The dataset is generated from **high-fidelity OrcaFlex simulations** of the IEA 10 MW offshore wind turbine in the hammerhead configuration under irregular wave loading.

Due to size constraints, the dataset is distributed separately.

### Dataset Setup
1. Download the dataset archive:
tensors.zip


2. Extract it to the **root directory** of this repository so the structure is:
```
Offshore_Wind_AE/
├── tensors/
├── run_curriculum.py
├── ...
```


The dataset is stored using NumPy memory-mapped arrays for efficient loading.

---

## How to Run

### Requirements
- Python 3.9+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- pandas

(Exact versions are flexible; standard scientific Python stack is sufficient.)

### Run Training
From the repository root:
python3 run_curriculum.py

### Outputs
- **Phase-wise training plots**  
  Generated dynamically in:
```
/Phase1/
/Phase2/
/Phase3/
```


- **Combined curriculum plots**  
Saved in:
/output_plots/



- **Random reconstruction examples**  
Saved in:
/random_plots/



- **Final summary statistics**  
Saved in:
/error_summary.txt



---

## Notes on Training

- The AE is trained using **curriculum learning**:
- **Phase 1**: Time-domain, FFT, adversarial, amplitude, offset, regularization
- **Phase 2**: STFT loss added
- **Phase 3**: Physics-based loss added

- Adversarial and physics losses are **explicitly capped** relative to reconstruction losses to ensure stability.

- Latent regularization enforces approximately zero-mean, unit-variance latents for diffusion compatibility.

---

## How to Cite

If you use this code or dataset, please cite the associated paper:

```
### BibTeX
@article{Dunton2025PhysicsInformedAE,
    title = {A Physics-Informed Autoencoder Framework for Digital Twin Modeling of Tower-Top Motions During Offshore Wind Turbine Installation},
    author = {Dunton, Dyllon and Bhaskaran, Saravanan and Zhu, Yifeng and Goupee, Andrew and Verma, Amrit},
    journal = {Springer Journal (Special Issue)},
    year = {2025}
}
```

---

## Funding

This research was supported by the **U.S. National Science Foundation (NSF)**  
Award No. **2343210 (CBET)**.

---

## License

This repository is intended for **academic and research use**.  
Please contact the authors before using this work for commercial purposes.

---

## Contact

For questions or collaboration inquiries:

**Dyllon Dunton**  
dyllon.dunton@maine.edu  
University of Maine
