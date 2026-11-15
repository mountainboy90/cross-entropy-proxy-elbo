"""Shared bootstrap for the stage scripts: puts the package on sys.path and parses --dataset."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxyelbo.config import get_config  # noqa: E402


def parse(description, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--dataset", choices=["nf1", "bcw"], default="nf1")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, get_config(args.dataset)
