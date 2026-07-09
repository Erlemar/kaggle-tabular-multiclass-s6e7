"""
submit.py — validate submission.csv against sample_submission before spending a slot.

    python templates/submit.py --check
    python templates/submit.py --check --submit -m "blend_v3: cv 0.4123"

Validation is free; a wasted daily submission slot is not. Always --check first.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_p = Path(__file__).resolve()
for _a in [_p, *_p.parents]:
    if (_a / "src" / "common.py").exists():
        sys.path.insert(0, str(_a / "src")); break
import common as C


def check(sub_path: Path) -> bool:
    sub = pd.read_csv(sub_path)
    ss = C.load_sample_submission()
    ok = True

    def bad(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL: {msg}")

    if list(sub.columns) != list(ss.columns):
        bad(f"columns/order differ.\n    sub : {list(sub.columns)}\n    want: {list(ss.columns)}")
    if len(sub) != len(ss):
        bad(f"row count {len(sub)} != sample_submission {len(ss)}")

    if C.ID in sub.columns and C.ID in ss.columns:
        s_ids, ss_ids = set(sub[C.ID]), set(ss[C.ID])
        if s_ids != ss_ids:
            miss, extra = ss_ids - s_ids, s_ids - ss_ids
            bad(f"id mismatch: {len(miss)} missing, {len(extra)} extra")
        if sub[C.ID].duplicated().any():
            bad("duplicate ids")

    val_cols = [c for c in sub.columns if c != C.ID]
    vals = sub[val_cols]
    if vals.select_dtypes("number").shape[1] == len(val_cols):  # numeric (prob) submission
        if not np.isfinite(vals.to_numpy()).all():
            bad("NaN/inf in values")
        if (vals.to_numpy() < -1e-9).any() or (vals.to_numpy() > 1 + 1e-9).any():
            bad("probabilities outside [0,1]")
        if len(val_cols) > 1:
            row_sums = vals.to_numpy().sum(1)
            if not np.allclose(row_sums, 1.0, atol=1e-3):
                bad(f"rows don't sum to 1 (min {row_sums.min():.4f}, max {row_sums.max():.4f})")
    else:  # label submission
        if vals.isna().any().any():
            bad("NaN labels")

    print("  PASS: submission matches sample_submission" if ok else "  -> fix before submitting")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(C.ROOT / "submission.csv"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("-m", "--message", default="submission")
    args = ap.parse_args()

    sub_path = Path(args.file)
    if not sub_path.exists():
        print(f"no submission at {sub_path} — run the blender first.")
        sys.exit(1)

    print(f"checking {sub_path} ...")
    ok = check(sub_path)
    if not ok:
        sys.exit(2)

    if args.submit:
        slug = C.CFG["competition"]
        cmd = ["kaggle", "competitions", "submit", "-c", slug, "-f", str(sub_path), "-m", args.message]
        print("submitting:", " ".join(cmd))
        try:
            print(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
            print("note: record CV -> LB in EXPERIMENTS_LOG.md once the score appears.")
        except Exception as e:
            print(f"submit failed ({e}). If this is a kernel-only comp, submit via the Kaggle UI instead.")


if __name__ == "__main__":
    main()
