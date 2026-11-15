"""Stage 8: every LaTeX table in the paper (results/<dataset>/tables/tab_*.tex), built from
the CSVs written by stages 2-6. For BCW it also writes appendix_table_rows_bcw.tex.
"""
import numpy as np
import pandas as pd

from _common import parse
from proxyelbo.utils import load_json

args, cfg = parse(__doc__)
ds, out, tab = args.dataset, cfg["OUT"], cfg["TAB"]
print(f"[{cfg['NAME']}] tables")

# ANOVA F-scores from Sharafi et al. (2025) Table 2, as mapped in the paper's Table 7
SOURCE_F = {"Lisch nodules": 1.95, "Hypertension": 1.48, "Skeletal dysplasia": 1.13,
            "Optic glioma": 1.09, "Astrocytoma": 0.95, "Hamartoma": 0.85, "Tumour case": 0.80,
            "Age of father": 0.77, "Axillary freckles": 0.68, "Dermal neurofibromas": 0.67,
            "Café-au-lait spots": 0.54, "Plexiform neurofibromas": 0.41,
            "Age at first diagnosis": 0.31, "Age of mother": 0.05, "Scoliosis": 0.02}


def num(x, nd=3, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:.{nd}f}"
    if signed and x >= 0:
        s = r"\phantom{-}" + s
    return f"${s}$"


def ci(lo, hi, nd=3):
    s = f"[{lo:.{nd}f}, {hi:.{nd}f}]"
    s = s.replace("[", "[").replace(", ", ",\\,")
    return f"$\\mathbf{{{s}}}$" if (lo > 0 or hi < 0) else f"${s}$"


def tex(name, caption, label, colspec, header, body, size=""):
    lines = [r"\begin{table}", rf"\tbl{{{caption}}}", "{" + size + rf"\begin{{tabular}}{{{colspec}}}\toprule",
             header + r" \\\midrule"] + [r + r" \\" for r in body] + [r"\bottomrule", r"\end{tabular}}",
                                                                       rf"\label{{{label}}}", r"\end{table}"]
    (tab / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote", f"{name}.tex")


def fmt_row(*cells):
    return " & ".join(str(c) for c in cells)


NOTATION = [
    ("$X$, $y$, $n$, $p$", "Covariate matrix, binary response, sample size, number of covariates"),
    (r"$\mathcal M$, $p^M$", "The black-box ensemble and its predicted probabilities"),
    ("$Y_1$", "The black box's thresholded hard label (70th training-set percentile of $p^M$)"),
    (r"$\mathcal S$, $\beta$, $p^s(\beta)$", "The logistic surrogate, its coefficients and predicted probabilities"),
    (r"$H(\cdot,\cdot)$", "Cross-entropy"),
    (r"$\delta(\beta)$", "Surrogate closeness, $H(y,p^s(\beta)) - H(y,p^M)$"),
    (r"$q(\beta)$, $\pi(\beta)$, $\Phi$", "Variational distribution, prior, variational family"),
    (r"$\mathcal L(q)$, $\widetilde{\mathcal L}(q)$", "Standard ELBO and the proposed cross-entropy proxy ELBO"),
    (r"$\tau^2$, $s_1$, $s_2$, $a,b,c,d$", "Prior variance and its Inverse-Gamma hyperprior parameters"),
    (r"$\eta$, $\kappa$, $m$", "MALA step size, MH proposal ridge, number of posterior draws"),
    (r"$w$, $\alpha$, $\mu_N$, $\Sigma_N$, $a_N$, $b_N$, $\xi_n$", "Variational Bayes quantities (Appendix B)"),
    (r"$\phi_j$", "SHAP value attributed to feature $j$"),
]

tag = "" if ds == "nf1" else "bcw-"
if ds == "nf1":
    tex("tab_notation", "Notation used throughout the paper.", "tab:notation", "ll", "Symbol & Meaning",
        [fmt_row(a, b) for a, b in NOTATION], size=r"\footnotesize")
dsname = "NF1" if ds == "nf1" else "Breast Cancer Wisconsin"

# ---------------- ensemble performance ----------------
perf = pd.read_csv(tab / "table_ensemble_performance.csv")
if ds == "nf1":
    keep = (perf.split == "Test") | (perf.model == "Stacked Ensemble")
else:
    keep = perf.model != "Soft Voting (RF+XGB+MLP)"
body = [fmt_row(r.model, r.split, *[f"{r[c]:.3f}" for c in ("acc", "spec", "recall", "logloss", "auc")])
        for _, r in perf[keep].iterrows()]
tex("tab_ensemble_perf", f"Base learner and ensemble performance, {dsname}.", f"tab:{tag}ensemble-perf",
    "lcccccc", "Model & Split & Acc. & Spec. & Recall & Log-loss & AUC", body)

# ---------------- surrogate performance ----------------
sp = pd.read_csv(tab / "table_surrogate_performance.csv")
if ds == "nf1":
    s = sp[sp.judged_against == "y"]
    body = [fmt_row(r.surrogate, r.split, f"{r.acc:.3f}", f"{r.auc:.3f}", f"{r.logloss:.3f}")
            for _, r in s.iterrows()]
    tex("tab_surrogate_perf", "Surrogate performance judged against the actual target $y$, NF1 data.",
        "tab:surrogate-perf", "lcccc", "Surrogate & Split & Acc. & AUC & Log-loss", body)
else:
    body = [fmt_row(r.surrogate, r.split, "$Y_1$" if r.judged_against == "Y1" else "$y$",
                    f"{r.acc:.3f}", f"{r.logloss:.3f}", f"{r.auc:.3f}") for _, r in sp.iterrows()]
    tex("tab_surrogate_perf", "Surrogate performance judged against $Y_1$ and $y$, Breast Cancer Wisconsin.",
        "tab:bcw-surrogate-perf", "llllcc", "Surrogate & Split & Judged against & Acc. & Log-loss & AUC", body)

# ---------------- coefficients (full table / appendix rows) ----------------
coef = pd.read_csv(tab / "table_coefficients.csv")
shp_f = tab / "table_shap_importance.csv"
shp = pd.read_csv(shp_f) if shp_f.exists() else pd.DataFrame(dict(feature=coef.feature, mean_abs_shap=np.nan,
                                                                  shap_slope=np.nan, shap_rank=np.nan))
cs = coef.merge(shp, on="feature")
rows = [fmt_row(r.feature, num(r.standard_lr, 2 if ds == "bcw" else 3), num(r.mh_mean), ci(r.mh_lo, r.mh_hi),
                num(r.mala_mean), ci(r.mala_lo, r.mala_hi), f"{r.mean_abs_shap:.4f}", num(r.shap_slope, 4))
        for _, r in cs.iterrows()]
header = (r"Feature & Standard LR & \multicolumn{2}{c}{Bayesian LR (MH-MCMC)} & \multicolumn{2}{c}{MALA} & Mean & SHAP \\"
          r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} & (MLE) & Mean & 95\% CI & Mean & 95\% CI & $|$SHAP$|$ & slope")
tex("tab_coefs_full", f"Coefficients, 95\\% credible intervals, mean $|$SHAP$|$ and SHAP slope, {dsname}. "
    "Credible intervals not spanning zero in bold.", f"tab:{'coefs' if ds == 'nf1' else 'bcw-appendix-full'}",
    "lccccccc", header, rows, size=r"\footnotesize")
if ds == "bcw":
    (tab / "appendix_table_rows_bcw.tex").write_text("\n".join(r + r" \\" for r in rows), encoding="utf-8")
    top = cs.sort_values("shap_rank").head(11)
    body = [fmt_row(r.feature, num(r.standard_lr, 2), f"{r.mh_mean:.3f}", f"{r.mala_mean:.3f}")
            for _, r in top.iterrows()]
    tex("tab_bcw_coefs", "Coefficients for the top-11 SHAP covariates, Breast Cancer Wisconsin.",
        "tab:bcw-coefs", "lccc", "Feature & Standard LR & Bayesian LR (MH-MCMC) & MALA", body)

# ---------------- representative credible intervals ----------------
sel = cs[cs.mh_credible ^ cs.mala_credible]
extra = ["Hypertension"] if ds == "nf1" else list(cs.sort_values("shap_rank").feature[:2])
sel = pd.concat([sel, cs[cs.feature.isin(extra) & ~cs.feature.isin(sel.feature)]])


def b(v, cred):
    return f"$\\mathbf{{{v:.3f}}}$" if cred else f"${v:.3f}$"


body = [fmt_row(r.feature, b(r.mh_lo, r.mh_credible), b(r.mh_hi, r.mh_credible),
                b(r.mala_lo, r.mala_credible), b(r.mala_hi, r.mala_credible)) for _, r in sel.iterrows()]
tex("tab_ci", f"Representative 95\\% credible intervals, {dsname}. Intervals not spanning zero in bold.",
    f"tab:{tag}ci", "lcccc",
    r"Feature & \multicolumn{2}{c}{Bayesian LR (MH-MCMC)} & \multicolumn{2}{c}{MALA} \\ & 2.5\% & 97.5\% & 2.5\% & 97.5\%",
    body)

# ---------------- alignment with source publication (NF1) ----------------
if ds == "nf1" and shp_f.exists():
    al = cs[cs.feature.isin(SOURCE_F)].copy()
    al["F"] = al.feature.map(SOURCE_F)
    al = al.sort_values("F", ascending=False).reset_index(drop=True)

    def cred(r):
        return {(True, True): "both", (True, False): "Bayesian LR only",
                (False, True): "MALA only", (False, False): "neither"}[(r.mh_credible, r.mala_credible)]

    body = [fmt_row(r.feature, cred(r), int(r.shap_rank), f"{r.F:.2f}", f"{i + 1}")
            for i, r in al.iterrows()]
    al.assign(credible_under=[cred(r) for _, r in al.iterrows()], source_rank=np.arange(1, len(al) + 1)
              )[["feature", "credible_under", "shap_rank", "F", "source_rank"]].to_csv(
        tab / "table_alignment_source.csv", index=False)
    tex("tab_alignment", "Surrogate credibility, SHAP rank, and the source publication's ANOVA F-score.",
        "tab:alignment", "lcccc", "Feature & Credible under & SHAP rank (of 18) & Source F-score & Source rank (of 15)",
        body, size=r"\footnotesize")

# ---------------- success criteria ----------------
af = tab / "table_shap_alignment.csv"
if af.exists():
    al = pd.read_csv(af)
    ll = sp[(sp.split == "Test") & (sp.judged_against == "y")].set_index("surrogate").logloss
    best = ll.min()
    rows, body = [], []
    for _, r in al.iterrows():
        ok_ll, ok_dir = ll[r.surrogate] <= 1.5 * best, r.alignment > 0.5
        verdict = "Success" if ok_ll and ok_dir else (
            "Failure (poorly calibrated)" if not ok_ll else "Failure (directionally wrong)")
        rows.append(dict(surrogate=r.surrogate, logloss_vs_y=ll[r.surrogate], alignment=r.alignment,
                         n_defined=r.n_defined, verdict=verdict))
        body.append(fmt_row(r.surrogate, f"{ll[r.surrogate]:.3f}", f"{100 * r.alignment:.1f}\\%", verdict))
    pd.DataFrame(rows).to_csv(tab / "table_success_criteria.csv", index=False)
    tex("tab_success", f"Surrogate success criteria on {dsname} (alignment over {int(al.n_defined[0])} covariates).",
        f"tab:success-{ds}", "lccl", r"Surrogate & Log-loss vs.\ $y$ & SHAP alignment & Verdict", body)

# ---------------- step size ----------------
f = tab / "table_mala_step_size_sensitivity.csv"
if f.exists():
    st = pd.read_csv(f)
    body = [fmt_row(f"{r.eta:g}" + (" (used)" if r.used else ""), f"{r.acceptance:.1f}", f"{r.mean_post_sd:.3f}")
            for _, r in st.iterrows()]
    tex("tab_stepsize", f"MALA step-size sensitivity, {dsname} (3,000-iteration trial chains).",
        f"tab:{tag}stepsize", "ccc", r"$\eta$ & Acceptance (\%) & Mean posterior coef. s.d.", body)

# ---------------- chain-length scaling ----------------
f = tab / "table_mcmc_length_scaling.csv"
if f.exists():
    sc = pd.read_csv(f)
    body = [fmt_row(f"{r.m:,}".replace(",", "{,}"), r.method,
                    *[f"{r[c]:.3f}" for c in ("auc_test_vs_Y1", "auc_test_vs_y", "acc_test_vs_Y1", "acc_test_vs_y",
                                              "logloss_test_vs_Y1", "logloss_test_vs_y", "coef_rmse_vs_max")])
            for _, r in sc.iterrows()]
    tex("tab_sim_scaling", f"MCMC chain-length scaling, {dsname}, test set judged against $Y_1$ and $y$.",
        f"tab:{tag}sim-scaling", "lcccccccc",
        r"$m$ & Method & AUC vs.\ $Y_1$ & AUC vs.\ $y$ & Acc.\ vs.\ $Y_1$ & Acc.\ vs.\ $y$ & "
        r"Log-loss vs.\ $Y_1$ & Log-loss vs.\ $y$ & Coef.\ RMSE vs.\ $m{=}100$k", body, size=r"\footnotesize")

# ---------------- multi-chain diagnostics ----------------
f = tab / "table_multichain_diagnostics.csv"
if f.exists():
    dg = pd.read_csv(f).groupby("method").agg(max_rhat=("rhat", "max"), min_ess=("ess_bulk", "min")).reset_index()
    dg.to_csv(tab / "table_multichain_summary.csv", index=False)
    body = [fmt_row(r.method, f"{r.max_rhat:.3f}", f"{r.min_ess:,.0f}") for _, r in dg.iterrows()]
    tex("tab_rhat", f"Four-chain convergence diagnostics, {dsname}.", f"tab:{tag}rhat", "lcc",
        r"Sampler & max $\hat R$ & min bulk ESS", body)

f = tab / "table_pymc_crosscheck.csv"
if f.exists():
    pc = pd.read_csv(f)
    body = [fmt_row(r.sampler, "$Y_1$" if r.target == "Y1" else "$y$", f"{r.corr_posterior_means:.3f}",
                    f"{r.pymc_max_rhat:.3f}") for _, r in pc.iterrows()]
    tex("tab_pymc", f"PyMC/NUTS cross-check of the hand-coded samplers, {dsname}.", f"tab:{tag}pymc", "llcc",
        r"Sampler & Target & Corr.\ of posterior means & PyMC max $\hat R$", body)

# ---------------- variational Bayes ----------------
f = tab / "table_drugowitsch_vb_coefficients.csv"
if f.exists():
    vb = pd.read_csv(f).merge(coef, on="feature")
    ea = load_json(out / "vb_summary.json")["E_alpha"]
    if ds == "nf1":
        body = [fmt_row(r.feature, num(r.standard_lr), num(r.mh_mean), num(r.mala_mean),
                        num(r.vb_mean, 4), f"${r.vb_sd:.4f}$") for _, r in vb.iterrows()]
        tex("tab_vb", f"VB posterior (mean, s.d.) for all NF1 covariates. $\\mathbb E[\\alpha]\\approx{ea:,.1f}$.",
            "tab:vb-nf1", "lccccc", r"Feature & Standard LR & Bayesian LR (MH-MCMC) & MALA & VB mean & VB s.d.",
            body, size=r"\footnotesize")
    else:
        top = vb.reindex(vb.vb_mean.abs().sort_values(ascending=False).index).head(12)
        body = [fmt_row(r.feature, num(r.vb_mean), f"${r.vb_sd:.3f}$") for _, r in top.iterrows()]
        tex("tab_vb", f"VB posterior for the twelve largest-magnitude covariates, BCW. $\\mathbb E[\\alpha]\\approx{ea:.3f}$.",
            "tab:vb-bcw", "lcc", r"Feature & VB mean & VB s.d.", body)
