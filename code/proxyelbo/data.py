"""Data loading and preparation (Section 4.1, Appendix A).

NF1 : UCI "Neurofibromatosis Type 1 clinical symptoms" (dataset-uci.xlsx), n=296, 18 covariates.
BCW : UCI "Breast Cancer Wisconsin (Diagnostic)" (wdbc.data), n=569, 30 covariates.

Both are split 70/30 stratified on y (seed 2024). Missing values are imputed with the
training median and continuous covariates are standardised with training moments.
"""
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, PROJECT_DIR

# raw column -> paper display name (Table 4)
NF1_COLUMNS = {
    "Age of Mother": "Age of mother",
    "Age of Father": "Age of father",
    "Age at First Diagnosis": "Age at first diagnosis",
    "Tumour Case": "Tumour case",
    "CAFE": "Café-au-lait spots",
    "Axillary Freckles": "Axillary freckles",
    "Inguinal Freckles": "Inguinal freckles",
    "Lisch Nodules": "Lisch nodules",
    "Dermal Neurofibromins": "Dermal neurofibromas",
    "Plexiform Neurofibromins": "Plexiform neurofibromas",
    "Optic Glioma": "Optic glioma",
    "Skeletal Dysplasia": "Skeletal dysplasia",
    "Learning Disability": "Learning disability",
    "Hypertension": "Hypertension",
    "Astrocytoma": "Astrocytoma",
    "Hamartoma": "Hamartoma",
    "Scoliosis": "Scoliosis",
    "Other Symptoms": "Other symptoms",
}
NF1_CONTINUOUS = ["Age of mother", "Age of father", "Age at first diagnosis"]

BCW_BASE = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
            "concavity", "concave points", "symmetry", "fractal dimension"]
BCW_COLUMNS = ([f"mean {b}" for b in BCW_BASE] + [f"{b} error" for b in BCW_BASE]
               + [f"worst {b}" for b in BCW_BASE])


def _ensure_extracted(zip_name, member, subdir):
    target = DATA_DIR / subdir / member
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DATA_DIR / zip_name) as z:
            z.extract(member, target.parent)
    return target


def load_nf1_raw():
    path = _ensure_extracted(
        "neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases.zip",
        "dataset-uci.xlsx", "nf1")
    df = pd.read_excel(path, sheet_name="Dataset")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    # the cafe-au-lait header has a mangled accent in the xlsx; match it by prefix
    df = df.rename(columns={c: "CAFE" for c in df.columns if str(c).startswith("Caf")})
    y = df["Case Type"].astype(int).to_numpy()          # 1 = familial, 0 = sporadic
    X = df[list(NF1_COLUMNS)].rename(columns=NF1_COLUMNS)
    return X, y


def load_bcw_raw():
    path = _ensure_extracted("breast+cancer+wisconsin+diagnostic.zip", "wdbc.data", "bcw")
    df = pd.read_csv(path, header=None)
    y = (df[1] == "M").astype(int).to_numpy()            # 1 = malignant, 0 = benign
    X = df.iloc[:, 2:].astype(float)
    X.columns = BCW_COLUMNS
    return X, y


def prepare(dataset, cfg):
    """Return dict with raw and standardised train/test frames plus targets."""
    if dataset == "nf1":
        X, y = load_nf1_raw()
        cont = NF1_CONTINUOUS
    else:
        X, y = load_bcw_raw()
        cont = list(X.columns)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=cfg["TEST_SIZE"], stratify=y, random_state=cfg["SPLIT_SEED"])
    Xtr_raw = Xtr_raw.reset_index(drop=True)
    Xte_raw = Xte_raw.reset_index(drop=True)

    med = Xtr_raw.median()
    Xtr_raw = Xtr_raw.fillna(med)
    Xte_raw = Xte_raw.fillna(med)

    mu, sd = Xtr_raw[cont].mean(), Xtr_raw[cont].std(ddof=0).replace(0, 1.0)
    Xtr, Xte = Xtr_raw.copy(), Xte_raw.copy()
    Xtr[cont] = (Xtr[cont] - mu) / sd
    Xte[cont] = (Xte[cont] - mu) / sd
    return dict(Xtr=Xtr, Xte=Xte, Xtr_raw=Xtr_raw, Xte_raw=Xte_raw,
                ytr=np.asarray(ytr), yte=np.asarray(yte), features=list(X.columns))


def save_prepared(d, out):
    for split in ("tr", "te"):
        df = d[f"X{split}"].copy()
        df["y"] = d[f"y{split}"]
        df.to_csv(out / f"{'train' if split == 'tr' else 'test'}_data.csv", index=False)
        d[f"X{split}_raw"].to_csv(out / f"{'train' if split == 'tr' else 'test'}_data_raw.csv",
                                  index=False)


def load_prepared(out):
    tr = pd.read_csv(out / "train_data.csv")
    te = pd.read_csv(out / "test_data.csv")
    ytr, yte = tr.pop("y").to_numpy(), te.pop("y").to_numpy()
    return dict(Xtr=tr, Xte=te, ytr=ytr, yte=yte, features=list(tr.columns),
                Xtr_raw=pd.read_csv(out / "train_data_raw.csv"),
                Xte_raw=pd.read_csv(out / "test_data_raw.csv"))
