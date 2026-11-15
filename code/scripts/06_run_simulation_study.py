"""Stage 6: simulation study on the real, fixed data (Section 9, App. A).

  (a) MALA step-size sweep on 3,000-iteration trial chains     -> table_mala_step_size_sensitivity.csv
  (b) chain-length scaling, m in {500 ... 100,000}, 20% burn-in -> table_mcmc_length_scaling.csv
  (c) 4 dispersed chains per sampler, split R-hat / bulk ESS    -> table_multichain_diagnostics.csv
  (d) optional PyMC/NUTS cross-check of both posteriors         -> table_pymc_crosscheck.csv

Use --only a,b,c,d to run a subset.
"""
import numpy as np
import pandas as pd

from _common import parse
from proxyelbo.data import load_prepared
from proxyelbo.diagnostics import multichain_summary
from proxyelbo.samplers import mala, mh_mcmc, summarize_chain
from proxyelbo.utils import load_json, model_eval, save_json, sigmoid, threshold_from_train

args, cfg = parse(__doc__, lambda ap: ap.add_argument("--only", default="a,b,c,d"))
parts = set(args.only.split(","))
out, tab = cfg["OUT"], cfg["TAB"]
d = load_prepared(out)
feats = d["features"]
Xtr, Xte = d["Xtr"].to_numpy(float), d["Xte"].to_numpy(float)
ytr, yte = d["ytr"], d["yte"]
pM_tr = np.load(out / "pred_prob_train.npy")
Y1tr, Y1te = np.load(out / "Y1_train.npy"), np.load(out / "Y1_test.npy")
hp = load_json(out / "surrogate_summary.json")["hyperprior"]
seed, eta = cfg["MCMC_SEED"], cfg["MALA_ETA"]
p = Xtr.shape[1]
print(f"[{cfg['NAME']}] simulation study: {sorted(parts)}")

# ---------------- (a) step-size sensitivity ----------------
if "a" in parts:
    m = cfg["STEPSIZE_ITERS"]
    burn = int(cfg["SCALING_BURN_FRAC"] * m)
    rows = []
    for e in cfg["STEPSIZE_GRID"]:
        r = mala(ytr, Xtr, hp, e, m, seed, p_M=pM_tr, verbose=False)
        rows.append(dict(eta=e, acceptance=100 * r["accept"],
                         mean_post_sd=float(r["beta"][burn:].std(0, ddof=1).mean()),
                         used=(e == eta)))
        print(f"  eta={e:<8g} acceptance {100 * r['accept']:5.1f}%  spread {rows[-1]['mean_post_sd']:.3f}")
    pd.DataFrame(rows).to_csv(tab / "table_mala_step_size_sensitivity.csv", index=False)

# ---------------- (b) chain-length scaling ----------------
if "b" in parts:
    res, means = [], {}
    for m in cfg["SCALING_M"]:
        burn = int(cfg["SCALING_BURN_FRAC"] * m)
        for method, fn in (("MH-MCMC", lambda: mh_mcmc(Y1tr, Xtr, hp, m, seed, cfg["MCMC_RIDGE"], verbose=False)),
                           ("MALA", lambda: mala(ytr, Xtr, hp, eta, m, seed, p_M=pM_tr, verbose=False))):
            r = fn()
            b = r["beta"][burn:].mean(0)
            means[(m, method)] = b
            ptr, pte = sigmoid(Xtr @ b), sigmoid(Xte @ b)
            thr = threshold_from_train(ptr, cfg["Y1_PERCENTILE"])
            row = dict(m=m, method=method, runtime_s=r["runtime"], acceptance=r["accept"])
            for split, pp, yy, yy1 in (("train", ptr, ytr, Y1tr), ("test", pte, yte, Y1te)):
                for tgt, lab in (("y", yy), ("Y1", yy1)):
                    e_ = model_eval(lab, pp, thr)
                    for k in ("auc", "acc", "recall", "logloss"):
                        row[f"{k}_{split}_vs_{tgt}"] = e_[k]
            res.append(row)
            print(f"  m={m:>6} {method:<7} test logloss vs y {row['logloss_test_vs_y']:.3f} "
                  f"({r['runtime']:.1f}s)")
    mmax = max(cfg["SCALING_M"])
    for row in res:
        ref = means[(mmax, row["method"])]
        row["coef_rmse_vs_max"] = float(np.sqrt(np.mean((means[(row["m"], row["method"])] - ref) ** 2)))
    pd.DataFrame(res).to_csv(tab / "table_mcmc_length_scaling.csv", index=False)

# ---------------- (c) multi-chain R-hat / ESS ----------------
if "c" in parts:
    m, burn = cfg["M_ITER"], cfg["BURN_IN"]
    rng = np.random.default_rng(seed)
    starts = [np.zeros(p)] + [rng.standard_normal(p) for _ in range(cfg["N_CHAINS"] - 1)]
    chains = {"MH-MCMC": [], "MALA": []}
    for k, s0 in enumerate(starts):
        chains["MH-MCMC"].append(mh_mcmc(Y1tr, Xtr, hp, m, seed + k, cfg["MCMC_RIDGE"], beta0=s0,
                                         verbose=False)["beta"])
        chains["MALA"].append(mala(ytr, Xtr, hp, eta, m, seed + k, p_M=pM_tr, beta0=s0,
                                   verbose=False)["beta"])
        print(f"  chain {k + 1}/{len(starts)} done")
    rows = []
    for method, cl in chains.items():
        for r in multichain_summary(cl, burn, feats):
            rows.append(dict(method=method, **r))
    diag = pd.DataFrame(rows)
    diag.to_csv(tab / "table_multichain_diagnostics.csv", index=False)
    print(diag.groupby("method").agg(max_rhat=("rhat", "max"), min_ess=("ess_bulk", "min")).round(3))
    np.savez_compressed(out / "multichain_means.npz",
                        **{k: np.array([c[burn:].mean(0) for c in v]) for k, v in chains.items()})

# ---------------- (d) PyMC / NUTS cross-check (optional) ----------------
if "d" in parts:
    try:
        import arviz as az
        import pymc as pm
    except ImportError:
        print("  PyMC/ArviZ not installed: skipping the NUTS cross-check (pip install pymc arviz)")
    else:
        burn = cfg["BURN_IN"]
        ch = np.load(out / "chains.npz")
        own = {"Y1": summarize_chain(ch["mh_beta"], burn)["mean"],
               "y": summarize_chain(ch["mala_beta"], burn)["mean"]}
        rows = []
        for tgt, lab in (("Y1", Y1tr), ("y", ytr)):
            with pm.Model():
                s1 = pm.InverseGamma("s1", alpha=hp["a"], beta=hp["b"])
                s2 = pm.InverseGamma("s2", alpha=hp["c"], beta=hp["d"])
                beta = pm.Normal("beta", 0, pm.math.sqrt(s1 * s2), shape=p)
                pm.Bernoulli("obs", logit_p=pm.math.dot(Xtr, beta), observed=lab)
                idata = pm.sample(2000, tune=1000, chains=4, cores=1, random_seed=seed,
                                  progressbar=False)
            pmean = idata.posterior["beta"].mean(("chain", "draw")).values
            rows.append(dict(target=tgt, sampler="MH-MCMC" if tgt == "Y1" else "MALA",
                             corr_posterior_means=float(np.corrcoef(pmean, own[tgt])[0, 1]),
                             pymc_max_rhat=float(az.rhat(idata)["beta"].max())))
        pd.DataFrame(rows).to_csv(tab / "table_pymc_crosscheck.csv", index=False)
        print(pd.DataFrame(rows).round(4).to_string(index=False))

save_json({"stages": sorted(parts)}, out / "simulation_done.json")
