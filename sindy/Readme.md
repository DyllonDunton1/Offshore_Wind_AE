# Reduced-Order Modeling and Coefficient Identification for Tower-Top Motions

This repository contains the data and scripts required to reproduce the reduced-order
model identification and coefficient parameterization presented in the associated paper.

## 1. Input Data
The primary input consists of tower-top motion time-series data (fore–aft and side–side
displacements) and the corresponding wave elevation for each simulation case.  
The data are organized by sea state, defined by:
- Significant wave height (Hs)
- Peak period (Tp)
- Wave direction (Theta)
- Random wave seed

A case mapping file is provided to identify the sea-state parameters corresponding to
each simulation case (e.g., Case1, Case2, Case3, etc.), enabling consistent indexing
between the motion data and governing environmental conditions.

## 2. Sparse System Identification (SINDy)
The script `sindy_batch.py` uses the tower-top motion data to perform sparse system
identification independently for each simulation case.

Running this script generates the file:
- `sindy_all_cases_coefficients.csv`

which contains the identified reduced-order model coefficients for all cases.

## 3. Coefficient Regression
The script `coefficient_regression.py` performs regression on the identified SINDy
coefficients to express them as explicit functions of the governing sea-state parameters
Hs, Tp, and Theta.


