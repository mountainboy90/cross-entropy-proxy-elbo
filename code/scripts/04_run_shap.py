"""Stage 4: Kernel SHAP of the full stacked pipeline (Section 10) and directional alignment
of each surrogate with the SHAP slope (Section 10.1, Tables 9 and 13).

Outputs: shap_values.npy, shap_eval_raw.csv, table_shap_importance.csv, table_shap_alignment.csv
"""
import warnings

import joblib
import numpy as np
import pandas as pd
import shap

from _common import parse
from proxyelbo.data import load_prepared

warnings.filterwarnings("ignore")
args, cfg = parse(__doc__)
out, tab = cfg["OUT"], cfg["TAB"]
d = load_prepared(out)
stack = joblib.load(cfg["MODELS"] / "ensemble.joblib")["stack"]
feats = d["features"]

Xev, Xev_raw = d["Xte"], d["Xte_raw"]
if cfg["SHAP_EVAL_N"] and cfg["SHAP_EVAL_N"] < len(Xev):
    idx = np.random.default_rng(cfg["ENSEMBLE_SEED"]).choice(len(Xev), cfg["SHAP_EVAL_N"], replace=False)
    Xev, Xev_raw = Xev.iloc[idx].reset_index(drop=True), Xev_raw.iloc[idx].reset_index(drop=True)


def f(X):
    return stack.predict_proba(pd.DataFrame(X, columns=feats))[:, 1]


bg = shap.kmeans(d["Xtr"].to_numpy(float), cfg["SHAP_BACKGROUND_K"])
expl = shap.KernelExplainer(f, bg)
print(f"[{cfg['NAME']}] Kernel SHAP on {len(Xev)} rows, nsamples={cfg['SHAP_NSAMPLES']}")
sv = np.asarray(expl.shap_values(Xev.to_numpy(float), nsamples=cfg["SHAP_NSAMPLES"], silent=True))
np.save(out / "shap_values.npy", sv)
Xev_raw.to_csv(out / "shap_eval_raw.csv", index=False)


def slope(x, s):
    x = np.asarray(x, float)
    if np.var(x) == 0:
        return np.nan                         # constant covariate: slope undefined
    return float(np.polyfit(x, s, 1)[0])


imp = pd.DataFrame(dict(feature=feats, mean_abs_shap=np.abs(sv).mean(0),
                        shap_slope=[slope(Xev_raw[c], sv[:, j]) for j, c in enumerate(feats)]))
imp["shap_rank"] = imp.mean_abs_shap.rank(ascending=False, method="first").astype(int)
imp.to_csv(tab / "table_shap_importance.csv", index=False)

# ---- directional alignment (signs of surrogate coef vs SHAP slope) ----
coef = pd.read_csv(tab / "table_coefficients.csv")
m = coef.merge(imp, on="feature")
defined = m.shap_slope.notna() & (m.shap_slope.abs() > 1e-10)
rows = []
for col, name in (("standard_lr", "Standard LR"), ("mh_mean", "Bayesian LR (MH-MCMC)"),
                  ("mala_mean", "MALA (CE-directed)")):
    agree = np.sign(m.loc[defined, col]) == np.sign(m.loc[defined, "shap_slope"])
    rows.append(dict(surrogate=name, n_defined=int(defined.sum()), n_aligned=int(agree.sum()),
                     alignment=float(agree.mean())))
pd.DataFrame(rows).to_csv(tab / "table_shap_alignment.csv", index=False)
print(imp.sort_values("shap_rank").round(4).to_string(index=False))
print(pd.DataFrame(rows).round(3).to_string(index=False))
