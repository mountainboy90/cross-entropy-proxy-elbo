"""Stacked RF + XGBoost + MLP ensemble with a logistic-regression meta-model (Section 4.2).

Each base learner is tuned by 5-fold stratified CV selecting on AUC; the stack is fit on
out-of-fold base-learner probabilities. Soft voting (unweighted mean) is the comparison.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

GRIDS = {
    "rf": dict(
        n_estimators=[100, 300, 500],
        max_depth=[3, 4, 5, None],
        min_samples_leaf=[1, 2, 4],
        class_weight=[None, "balanced"],
    ),
    "xgb": dict(
        n_estimators=[100, 300],
        max_depth=[1, 2, 3, 4],
        learning_rate=[0.01, 0.05, 0.1],
        subsample=[0.8, 1.0],
        reg_lambda=[1.0, 10.0],
    ),
    "mlp": dict(
        hidden_layer_sizes=[(16,), (32,), (64,), (32, 16)],
        alpha=[1e-4, 1e-2, 1.0],
        activation=["relu", "tanh"],
    ),
}


def base_estimator(name, seed):
    if name == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=1)
    if name == "xgb":
        return XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss")
    return MLPClassifier(random_state=seed, max_iter=2000)


def tune_base_learners(Xtr, ytr, cfg, log=print):
    """Grid-search each base learner; returns (best params dict, cv-results frames)."""
    cv = StratifiedKFold(cfg["CV_FOLDS"], shuffle=True, random_state=cfg["ENSEMBLE_SEED"])
    best, tables = {}, {}
    for name, grid in GRIDS.items():
        override = (cfg.get("PARAM_OVERRIDES") or {}).get(name)
        gs = GridSearchCV(base_estimator(name, cfg["ENSEMBLE_SEED"]), grid, scoring="roc_auc",
                          cv=cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        tables[name] = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
        if override:
            best[name] = dict(override)
            log(f"  {name}: CV best AUC {gs.best_score_:.3f} {gs.best_params_}; "
                f"using configured override {override}")
        else:
            best[name] = gs.best_params_
            log(f"  {name}: CV best AUC {gs.best_score_:.3f} {gs.best_params_}")
    return best, tables


def build_base(params, seed):
    out = {}
    for name, p in params.items():
        est = base_estimator(name, seed)
        est.set_params(**p)
        out[name] = est
    return out


def make_stack(params, cfg):
    seed = cfg["ENSEMBLE_SEED"]
    return StackingClassifier(
        estimators=list(build_base(params, seed).items()),
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(cfg["CV_FOLDS"], shuffle=True, random_state=seed),
        stack_method="predict_proba",
        n_jobs=1,
    )


def oof_stack_proba(Xtr, ytr, params, cfg):
    """Out-of-fold stacked-ensemble probabilities on the training set."""
    cv = StratifiedKFold(cfg["CV_FOLDS"], shuffle=True, random_state=cfg["ENSEMBLE_SEED"])
    return cross_val_predict(make_stack(params, cfg), Xtr, ytr, cv=cv, method="predict_proba",
                             n_jobs=-1)[:, 1]


def fit_stack(Xtr, ytr, params, cfg):
    seed = cfg["ENSEMBLE_SEED"]
    stack = make_stack(params, cfg)
    stack.fit(Xtr, ytr)
    # standalone refits of each base learner on the full training set
    fitted = {k: v.fit(Xtr, ytr) for k, v in build_base(params, seed).items()}
    return stack, fitted


class SoftVote:
    """Unweighted mean of the fitted base learners' probabilities."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        return np.column_stack([1 - p, p])
