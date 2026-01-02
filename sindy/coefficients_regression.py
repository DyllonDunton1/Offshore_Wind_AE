# Linear-only parametric fits for SINDy coefficients vs (Hs, Tp, Theta)
# - Reads: sindy_all_cases_coefficients.csv
# - Averages across seeds (per Hs, Tp, Theta)
# - Fits linear models (no squares, no cross terms) for: a_x, b_x, c_x, a_y, b_y, c_y
# - Prints R^2 and coefficients
# - Saves results to: linear_models_summary.csv (R^2 + intercept + coeffs) and
#                     linear_models_predictions.csv (per sea-state fitted vs predicted)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ============
# 1) Load data
# ============
# If you're running locally, set this to your file location.
csv_path = Path("sindy_all_cases_coefficients.csv")  # or Path("/mnt/data/sindy_all_cases_coefficients.csv")
df = pd.read_csv(csv_path)

# ==========================================================
# 2) Average across random seeds for each (Hs, Tp, Theta)
#    -> one stable target value per sea state
# ==========================================================
targets = ["a_x","b_x","c_x","a_y","b_y","c_y"]
df_avg = (
    df.groupby(["Hs","Tp","Theta"])[targets]
      .mean()
      .reset_index()
      .sort_values(["Hs","Tp","Theta"])
      .reset_index(drop=True)
)

# ==========================
# 3) Fit linear models: y ~ Hs + Tp + Theta
# ==========================
X = df_avg[["Hs","Tp","Theta"]].values
results = []
preds = df_avg[["Hs","Tp","Theta"]].copy()

for y_name in targets:
    y = df_avg[y_name].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Store predictions
    preds[f"{y_name}_true"] = y
    preds[f"{y_name}_pred_lin"] = y_pred

    # Gather coefficients
    row = {
        "target": y_name,
        "R2_linear": r2,
        "Intercept": model.intercept_,
        "coef_Hs": model.coef_[0],
        "coef_Tp": model.coef_[1],
        "coef_Theta": model.coef_[2],
    }
    results.append(row)

# =========================================
# 4) Print a readable summary to the console
# =========================================
summary = pd.DataFrame(results)
with pd.option_context("display.float_format", "{:,.6f}".format):
    print("\n=== Linear-only regression summary (no squares, no cross terms) ===")
    print(summary.sort_values("target").reset_index(drop=True))

# Pretty-print explicit formulas
def fmt_coef(v):
    # Compact formatting for readability
    return f"{v:+.6f}"

print("\n=== Explicit linear formulas ===")
for _, r in summary.sort_values("target").iterrows():
    eq = (f"{r['target']}(Hs,Tp,Theta) = "
          f"{r['Intercept']:+.6f} "
          f"{fmt_coef(r['coef_Hs'])}*Hs "
          f"{fmt_coef(r['coef_Tp'])}*Tp "
          f"{fmt_coef(r['coef_Theta'])}*Theta")
    print(eq)

# ====================================================
# 5) Save: summary (R2 + parameters) and predictions
# ====================================================
summary_path = Path("linear_models_summary.csv")
preds_path = Path("linear_models_predictions.csv")
summary.to_csv(summary_path, index=False)
preds.to_csv(preds_path, index=False)

print(f"\nSaved summary to: {summary_path.resolve()}")
print(f"Saved predictions to: {preds_path.resolve()}")

# ======================================
# 6) (Optional) Quick sanity-check metric
#     Mean Absolute Error per target
# ======================================
mae_rows = []
for y_name in targets:
    mae = np.mean(np.abs(preds[f"{y_name}_true"] - preds[f"{y_name}_pred_lin"]))
    mae_rows.append({"target": y_name, "MAE_linear": mae})
mae_df = pd.DataFrame(mae_rows)
with pd.option_context("display.float_format", "{:,.6f}".format):
    print("\n=== Mean Absolute Error (linear-only) ===")
    print(mae_df.sort_values("target").reset_index(drop=True))
