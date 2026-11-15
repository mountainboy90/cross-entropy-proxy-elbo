"""Run the full pipeline (stages 1-8) for one or both datasets.

    python run_all.py                    # both datasets, all stages
    python run_all.py --dataset nf1      # one dataset
    python run_all.py --from 3           # resume from stage 3

After a full run on both datasets, the figures used by the manuscript are copied from
results/<dataset>/figures into ../figures, which manuscript/manuscript.tex reads via \\graphicspath.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
ROOT = Path(__file__).resolve().parents[1]
STAGES = ["01_prepare_data.py", "02_fit_ensemble.py", "03_fit_surrogates.py", "04_run_shap.py",
          "05_run_drugowitsch_vb.py", "06_run_simulation_study.py", "07_make_figures.py",
          "08_make_tables.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["nf1", "bcw", "both"], default="both")
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--to", dest="stop", type=int, default=len(STAGES))
args = ap.parse_args()

for ds in (["nf1", "bcw"] if args.dataset == "both" else [args.dataset]):
    for i, script in enumerate(STAGES, 1):
        if not args.start <= i <= args.stop:
            continue
        t0 = time.time()
        print(f"\n=== [{ds}] stage {i}: {script} ===", flush=True)
        subprocess.run([sys.executable, script, "--dataset", ds], cwd=SCRIPTS, check=True)
        print(f"=== [{ds}] stage {i} finished in {time.time() - t0:.0f}s ===", flush=True)

# Collect the figures the manuscript uses into ../figures.
if args.dataset == "both" and args.stop == len(STAGES):
    tex = (ROOT / "manuscript" / "manuscript.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+\.png)\}", tex))
    (ROOT / "figures").mkdir(exist_ok=True)
    for png in (ROOT / "results").glob("*/figures/*.png"):
        if png.name in used:
            shutil.copy2(png, ROOT / "figures" / png.name)
    print(f"{len(used)} manuscript figures collected in", ROOT / "figures")
