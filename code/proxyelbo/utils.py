"""Shared metrics: cross-entropy (Eq. 1), sigmoid, and the evaluation metrics of Section 4.4."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-15


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def calc_ce(y, p):
    """H(y, p) = -(1/n) sum[y log p + (1-y) log(1-p)]  (Eq. 1)."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def safe_auc(y, p):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


def classify(p, threshold):
    return (np.asarray(p) > threshold).astype(int)


def threshold_from_train(p_train, q=0.70):
    """Decision threshold = q-th percentile of the model's own training predictions."""
    return float(np.quantile(p_train, q))


def model_eval(y, p, threshold):
    """Accuracy, specificity, recall, precision, F1, log-loss, AUC at a fixed threshold."""
    y = np.asarray(y).astype(int)
    yhat = classify(p, threshold)
    tp = int(np.sum((y == 1) & (yhat == 1)))
    tn = int(np.sum((y == 0) & (yhat == 0)))
    fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0)))
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(acc=(tp + tn) / len(y), spec=spec, recall=rec, precision=prec, f1=f1,
                logloss=calc_ce(y, p), auc=safe_auc(y, p), tp=tp, tn=tn, fp=fp, fn=fn)


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def load_json(path):
    with open(path) as f:
        return json.load(f)
