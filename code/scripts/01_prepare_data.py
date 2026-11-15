"""Stage 1: load, split 70/30 (stratified, seed 2024), impute with train medians, standardise.

Outputs: train_data.csv, test_data.csv (+ _raw versions), table_data_summary.csv
"""
import pandas as pd

from _common import parse
from proxyelbo.data import prepare, save_prepared

args, cfg = parse(__doc__)
d = prepare(args.dataset, cfg)
save_prepared(d, cfg["OUT"])

rows = []
for f in d["features"]:
    tr, te = d["Xtr_raw"][f], d["Xte_raw"][f]
    binary = set(pd.unique(pd.concat([tr, te]))) <= {0, 1}
    rows.append(dict(feature=f, binary=binary,
                     train_positive=int(tr.sum()) if binary else None,
                     test_positive=int(te.sum()) if binary else None,
                     train_mean=tr.mean(), test_mean=te.mean()))
pd.DataFrame(rows).to_csv(cfg["TAB"] / "table_data_summary.csv", index=False)

print(f"[{cfg['NAME']}] n_train={len(d['ytr'])} (pos {d['ytr'].sum()}), "
      f"n_test={len(d['yte'])} (pos {d['yte'].sum()}), p={len(d['features'])}")
