# scripts/eval_plag_aware.py
"""Collate the 3-way plag-aware evaluation from wandb.

Pulls per-class val APs for the baseline, encoder-only, and encoder+synthetic
fine-tuning runs and prints the spec's comparison table.

Usage:
  conda run -n crism python scripts/eval_plag_aware.py \\
    --baseline ft_bland_v3_lrscale0001_cont1 \\
    --enc_only ft_plag_aware_real_only \\
    --enc_synth ft_plag_aware_real_plus_synth
"""
import argparse

import wandb

PROJECT = "space-imagery-center/crism-mineral-classification"
CLASSES = ["olivine", "lcp", "hcp", "plagioclase", "other"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--enc_only", required=True)
    ap.add_argument("--enc_synth", required=True)
    args = ap.parse_args()

    api = wandb.Api()
    runs = {r.name: r for r in api.runs(PROJECT, per_page=200, order="-created_at")}

    rows = [("baseline", args.baseline), ("encoder-only", args.enc_only),
            ("encoder+synth", args.enc_synth)]
    hdr = f'{"run":>16s}  {"mAP":>7s}  ' + "  ".join(f"{c[:5]:>6s}" for c in CLASSES)
    print(hdr); print("-" * len(hdr))
    plag_baseline = None
    for label, name in rows:
        r = runs.get(name)
        if r is None:
            print(f"{label:>16s}  (run '{name}' not found)"); continue
        s = r.summary
        def g(k):
            return s.get(k, float("nan"))
        plag = g("val_AP_plagioclase")
        if label == "baseline":
            plag_baseline = plag
        cells = "  ".join(f"{g('val_AP_'+c):>6.3f}" for c in CLASSES)
        print(f"{label:>16s}  {g('val_mAP'):>7.4f}  {cells}")
    if plag_baseline == plag_baseline:  # not NaN
        print(f"\nplag baseline = {plag_baseline:.3f}; "
              f"publishable target = 0.60; signal gate = 0.20")


if __name__ == "__main__":
    main()
