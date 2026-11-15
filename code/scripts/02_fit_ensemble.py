"""Stage 2: tune and fit the stacked RF + XGBoost + MLP ensemble (Sections 4.2 and 6).

Outputs: pred_prob_{train,test}.npy, Y1_{train,test}.npy, base_probs.npz, ensemble.joblib,
         grid_search_{rf,xgb,mlp}.csv, table_ensemble_performance.csv
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from _common import parse
from proxyelbo.data import load_prepared
from proxyelbo.ensemble import SoftVote, fit_stack, oof_stack_proba, tune_base_learners
from proxyelbo.utils import load_json, model_eval, save_json, threshold_from_train

warnings.filterwarnings("ignore")
args, cfg = parse(__doc__, lambda ap: ap.add_argument("--retune", action="store_true",
                                                      help="redo the CV grid search"))
out = cfg["OUT"]
d = load_prepared(out)
Xtr, Xte, ytr, yte = d["Xtr"], d["Xte"], d["ytr"], d["yte"]

if args.retune or not Path(out / "ensemble_params.json").exists():
    print(f"[{cfg['NAME']}] grid search (5-fold stratified CV, AUC)")
    params, tables = tune_base_learners(Xtr, ytr, cfg)
    for k, t in tables.items():
        t.to_csv(out / f"grid_search_{k}.csv", index=False)
    save_json(params, out / "ensemble_params.json")
else:
    params = load_json(out / "ensemble_params.json")
    params = {k: {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
              for k, v in params.items()}
    print(f"[{cfg['NAME']}] reusing tuned parameters (pass --retune to redo the grid search)")
for k, v in (cfg.get("PARAM_OVERRIDES") or {}).items():
    params[k] = dict(v)          # paper-matched settings take precedence over the CV choice
print("  base-learner settings:", params)

stack, base = fit_stack(Xtr, ytr, params, cfg)
models = {"Random Forest": base["rf"], "XGBoost": base["xgb"], "MLP": base["mlp"],
          "Soft Voting (RF+XGB+MLP)": SoftVote(base), "Stacked Ensemble": stack}

# The stack's training-set p^M are out-of-fold predictions (the paper's honest train row);
# base learners report ordinary in-sample training fit.
oof = oof_stack_proba(Xtr, ytr, params, cfg)
rows, probs = [], {}
for name, mdl in models.items():
    ptr = oof if name == "Stacked Ensemble" else mdl.predict_proba(Xtr)[:, 1]
    pte = mdl.predict_proba(Xte)[:, 1]
    probs[name] = (ptr, pte)
    thr = threshold_from_train(ptr, cfg["Y1_PERCENTILE"])
    for split, y, pp in (("Train", ytr, ptr), ("Test", yte, pte)):
        rows.append(dict(model=name, split=split, threshold=thr, **model_eval(y, pp, thr)))
perf = pd.DataFrame(rows)
perf.to_csv(cfg["TAB"] / "table_ensemble_performance.csv", index=False)

pM_tr, pM_te = probs["Stacked Ensemble"]
thr = threshold_from_train(pM_tr, cfg["Y1_PERCENTILE"])
np.save(out / "pred_prob_train.npy", pM_tr)
np.save(out / "pred_prob_test.npy", pM_te)
np.save(out / "Y1_train.npy", (pM_tr > thr).astype(int))
np.save(out / "Y1_test.npy", (pM_te > thr).astype(int))
np.savez(out / "base_probs.npz", **{k.replace(" ", "_"): np.concatenate(v) for k, v in probs.items()})
joblib.dump(dict(stack=stack, base=base, params=params, threshold=thr), cfg["MODELS"] / "ensemble.joblib")

meta = stack.final_estimator_
print("  meta-model weights:", np.round(meta.coef_.ravel(), 3), "intercept", np.round(meta.intercept_, 3))
print(perf[["model", "split", "acc", "spec", "recall", "logloss", "auc"]].round(3).to_string(index=False))
