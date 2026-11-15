"""Stage 5: Drugowitsch mean-field VB, standard (JJ) ELBO vs proposed proxy ELBO (Section 5, App. B).

Outputs: table_drugowitsch_elbo_trace.csv, table_drugowitsch_vb_coefficients.csv, vb_summary.json
With --paper-kl the same files get a _paperkl suffix, so both versions can sit side by side.
"""
import numpy as np
import pandas as pd

from _common import parse
from proxyelbo.data import load_prepared
from proxyelbo.utils import save_json
from proxyelbo.vb import run_drugowitsch_vb

args, cfg = parse(__doc__, lambda ap: ap.add_argument(
    "--paper-kl", action="store_true",
    help="use Appendix B's printed KL (+p log E[alpha], a sign error) to recreate the printed Figure 1"))
out, tab = cfg["OUT"], cfg["TAB"]
d = load_prepared(out)
pM_tr = np.load(out / "pred_prob_train.npy")

r = run_drugowitsch_vb(d["Xtr"].to_numpy(float), d["ytr"], pM_tr, iters=cfg["VB_ITERS"],
                       mc_draws=cfg["VB_MC_DRAWS"], a0=cfg["VB_A0"], b0=cfg["VB_B0"],
                       seed=cfg["MCMC_SEED"], paper_kl=args.paper_kl)
sfx = "_paperkl" if args.paper_kl else ""
trace = pd.DataFrame(r["trace"])
trace.to_csv(tab / f"table_drugowitsch_elbo_trace{sfx}.csv", index=False)
pd.DataFrame(dict(feature=d["features"], vb_mean=r["mu"], vb_sd=r["sd"])
             ).to_csv(tab / f"table_drugowitsch_vb_coefficients{sfx}.csv", index=False)
corr = float(np.corrcoef(trace.elbo_standard, trace.elbo_proposed)[0, 1])
save_json(dict(paper_kl=args.paper_kl, E_alpha=r["E_alpha"], a_N=r["a_N"], b_N=r["b_N"], H_M_y=r["h_M"],
               corr_standard_vs_proposed=corr), out / f"vb_summary{sfx}.json")
print(f"[{cfg['NAME']}] {'paper (printed) KL' if args.paper_kl else 'correct KL'}: E[alpha]={r['E_alpha']:.3f}; final ELBO std={trace.elbo_standard.iloc[-1]:.2f}, "
      f"proposed={trace.elbo_proposed.iloc[-1]:.2f}; corr={corr:.3f}")
