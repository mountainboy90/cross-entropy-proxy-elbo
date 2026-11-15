"""Stage 3: Standard-LR benchmark vs y, hyperprior, and the three surrogates (Sections 3, 4.3, 7, 8).

  Standard LR            : MLE, no intercept, fit to Y1
  Bayesian LR (MH-MCMC)  : Algorithm 1, fit to Y1
  MALA (CE-directed)     : Algorithm 2, fit to y

Outputs: chains.npz, surrogate_summary.json, table_lr_benchmark_vs_y.csv,
         table_surrogate_performance.csv, table_coefficients.csv
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from _common import parse
from proxyelbo.data import load_prepared
from proxyelbo.samplers import build_hyperprior, mala, mh_mcmc, summarize_chain
from proxyelbo.utils import calc_ce, model_eval, save_json, sigmoid, threshold_from_train

warnings.filterwarnings("ignore")
args, cfg = parse(__doc__)
out, tab = cfg["OUT"], cfg["TAB"]
d = load_prepared(out)
feats = d["features"]
Xtr, Xte = d["Xtr"].to_numpy(float), d["Xte"].to_numpy(float)
ytr, yte = d["ytr"], d["yte"]
pM_tr, pM_te = np.load(out / "pred_prob_train.npy"), np.load(out / "pred_prob_test.npy")
Y1tr, Y1te = np.load(out / "Y1_train.npy"), np.load(out / "Y1_test.npy")
n, p = Xtr.shape
print(f"[{cfg['NAME']}] n={n}, p={p}, Y1 positives {Y1tr.sum()}/{n}")

# ---- Section 4.3: standalone LR vs y (statsmodels, p-values) ----
try:
    res = sm.Logit(ytr, Xtr).fit(disp=0, maxiter=200)
except Exception:
    res = sm.Logit(ytr, Xtr).fit(disp=0, method="bfgs", maxiter=5000)
p_bench_tr, p_bench_te = res.predict(Xtr), res.predict(Xte)
thr_b = threshold_from_train(p_bench_tr, cfg["Y1_PERCENTILE"])
pd.DataFrame(dict(feature=feats, coef=res.params, se=res.bse, z=res.tvalues, p_value=res.pvalues)
             ).to_csv(tab / "table_lr_benchmark_vs_y.csv", index=False)
bench = {s: model_eval(y, pp, thr_b) for s, y, pp in
         (("train", ytr, p_bench_tr), ("test", yte, p_bench_te))}
print(f"  LR-vs-y benchmark: train acc {bench['train']['acc']:.3f} AUC {bench['train']['auc']:.3f}; "
      f"test acc {bench['test']['acc']:.3f} AUC {bench['test']['auc']:.3f}")

# ---- Standard LR surrogate (fit to Y1) ----
lr = LogisticRegression(penalty=None, fit_intercept=False, max_iter=10000).fit(Xtr, Y1tr)
beta_lr = lr.coef_.ravel()
h_lr_y1 = calc_ce(Y1tr, sigmoid(Xtr @ beta_lr))
h_m_y = calc_ce(ytr, pM_tr)

# ---- Hyperprior (Eq. 15) ----
hp = build_hyperprior(n, p, h_lr_y1, h_m_y, cfg["CE_FLOOR"])
print(f"  H(Y1,p^LR)={h_lr_y1:.4f}  H(y,p^M)={h_m_y:.4f}  hyperprior={ {k: round(v, 3) for k, v in hp.items()} }")

# ---- Samplers ----
m, burn, seed = cfg["M_ITER"], cfg["BURN_IN"], cfg["MCMC_SEED"]
mh = mh_mcmc(Y1tr, Xtr, hp, m, seed, ridge=cfg["MCMC_RIDGE"])
ml = mala(ytr, Xtr, hp, cfg["MALA_ETA"], m, seed, p_M=pM_tr)
np.savez_compressed(out / "chains.npz", mh_beta=mh["beta"], mh_loglik=mh["loglik"],
                    mala_beta=ml["beta"], mala_grad=ml["grad"], mala_loglik=ml["loglik"],
                    mala_delta=ml["delta"], beta_lr=beta_lr)

s_mh, s_ml = summarize_chain(mh["beta"], burn), summarize_chain(ml["beta"], burn)

# ---- Surrogate performance (point estimate = posterior mean) ----
surr = {"Standard LR": beta_lr, "Bayesian LR (MH-MCMC)": s_mh["mean"], "MALA (CE-directed)": s_ml["mean"]}
rows = []
for name, b in surr.items():
    ptr, pte = sigmoid(Xtr @ b), sigmoid(Xte @ b)
    thr = threshold_from_train(ptr, cfg["Y1_PERCENTILE"])
    for split, pp, yy, yy1 in (("Train", ptr, ytr, Y1tr), ("Test", pte, yte, Y1te)):
        for target, lab in (("y", yy), ("Y1", yy1)):
            rows.append(dict(surrogate=name, split=split, judged_against=target,
                             **model_eval(lab, pp, thr)))
perf = pd.DataFrame(rows)
perf.to_csv(tab / "table_surrogate_performance.csv", index=False)

coef = pd.DataFrame(dict(
    feature=feats, standard_lr=beta_lr,
    mh_mean=s_mh["mean"], mh_sd=s_mh["sd"], mh_lo=s_mh["lo"], mh_hi=s_mh["hi"],
    mala_mean=s_ml["mean"], mala_sd=s_ml["sd"], mala_lo=s_ml["lo"], mala_hi=s_ml["hi"]))
coef["mh_credible"] = (coef.mh_lo > 0) | (coef.mh_hi < 0)
coef["mala_credible"] = (coef.mala_lo > 0) | (coef.mala_hi < 0)
coef.to_csv(tab / "table_coefficients.csv", index=False)

save_json(dict(hyperprior=hp, H_LR_Y1=h_lr_y1, H_M_y=h_m_y, H_M_y_test=calc_ce(yte, pM_te),
               accept_mh=mh["accept"], accept_mala=ml["accept"], eta=cfg["MALA_ETA"],
               runtime_mh=mh["runtime"], runtime_mala=ml["runtime"],
               mean_ci_width_mh=float((coef.mh_hi - coef.mh_lo).mean()),
               mean_ci_width_mala=float((coef.mala_hi - coef.mala_lo).mean()),
               delta_post_burn_mean=float(ml["delta"][burn:].mean()),
               lr_benchmark=bench),
          out / "surrogate_summary.json")

view = perf[perf.judged_against == "y"][["surrogate", "split", "acc", "auc", "logloss"]]
print(view.round(3).to_string(index=False))
print("  credible under MH  :", list(coef.feature[coef.mh_credible]))
print("  credible under MALA:", list(coef.feature[coef.mala_credible]))
