import numpy as np
import pandas as pd
import pysindy as ps
from pysindy.feature_library import IdentityLibrary
from tqdm import tqdm   # For progress bar

mapping_file = "Case_Map_1_to_7440.xlsx"
results_file = "results.xlsx"
dt = 0.1

# Load the case map
case_map = pd.read_excel(mapping_file)
assert case_map.shape[0] == 7440

out_records = []

def safe_float_array(arr):
    arr = np.array(arr)  # Ensure it's an array
    try:
        arr = arr.astype(float)
    except ValueError:
        arr = pd.to_numeric(arr, errors='coerce')
        arr = np.array(arr)
    return arr

for idx, row in tqdm(case_map.iterrows(), total=7440):
    case_idx = row['Case']    # Should be 1-based (Case1, Case2, ...)
    sheet_name = f"Case{case_idx}"
    Hs, Tp, Theta, Seed = row['Hs'], row['Tp'], row['Theta'], row['Seed']
    try:
        df = pd.read_excel(results_file, sheet_name=sheet_name)
        # Robust cleaning, coerce all columns to float and drop invalid rows
        x_disp = safe_float_array(df.iloc[:, 0].values)
        y_disp = safe_float_array(df.iloc[:, 1].values)
        eta    = safe_float_array(df.iloc[:, 2].values)
        valid = (~np.isnan(x_disp)) & (~np.isnan(y_disp)) & (~np.isnan(eta))
        if not np.all(valid):
            print(f"Warning: NaN in {sheet_name}. Dropping invalid time steps.")
        x_disp, y_disp, eta = x_disp[valid], y_disp[valid], eta[valid]
    except Exception as e:
        print(f"Skipped {sheet_name}: {e}")
        continue

    N = min(len(x_disp), len(y_disp), len(eta))
    if N < 5:
        continue

    x_disp, y_disp, eta = x_disp[:N], y_disp[:N], eta[:N]
    x_dot = np.gradient(x_disp, dt)
    y_dot = np.gradient(y_disp, dt)
    x_ddot = np.gradient(x_dot, dt)
    y_ddot = np.gradient(y_dot, dt)

    theta_rad = np.deg2rad(Theta)
    eta_x = eta * np.cos(theta_rad)
    eta_y = eta * np.sin(theta_rad)

    Theta_x = np.column_stack([x_disp, x_dot, eta_x])
    Theta_y = np.column_stack([y_disp, y_dot, eta_y])

    model_x = ps.SINDy(
        feature_library=IdentityLibrary(),
        optimizer=ps.STLSQ(threshold=1e-5),
        feature_names=["x", "xdot", "etax"]
    )
    model_x.fit(Theta_x, t=dt, x_dot=x_ddot.reshape(-1, 1))
    coefs_x = model_x.coefficients().flatten()

    model_y = ps.SINDy(
        feature_library=IdentityLibrary(),
        optimizer=ps.STLSQ(threshold=1e-5),
        feature_names=["y", "ydot", "etay"]
    )
    model_y.fit(Theta_y, t=dt, x_dot=y_ddot.reshape(-1, 1))
    coefs_y = model_y.coefficients().flatten()

    out_records.append([case_idx, Hs, Tp, Theta, Seed, *coefs_x, *coefs_y])

param_df = pd.DataFrame(
    out_records,
    columns=[
        "Case", "Hs", "Tp", "Theta", "Seed",
        "a_x", "b_x", "c_x",
        "a_y", "b_y", "c_y"
    ]
)
param_df.to_csv("sindy_all_cases_coefficients.csv", index=False)
print(param_df.head())
