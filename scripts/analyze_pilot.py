"""
Paired analysis for pilot_v2 results.

For each frame, delta = metric(attacked) - metric(clean), within one arm.
The test statistic is diff = delta_cmp - delta_ref on the same frame, i.e.
"does the hardened model respond to this attack differently than baseline?"
Two-sided throughout; the pilot shouldn't assume a direction.

Usage:
  python analyze_pilot.py --csv pilot_v2_out/results.csv --ref baseline_s0 --cmp mnat_s0
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logutil import start_log  # noqa: E402

META = {"arm", "seq", "frame", "attack", "eps_255"}
PRIMARY = ["stage4_rho_0.001", "stage4_svar", "stage4_chent", "n_dets"]


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    metrics = [k for k in rows[0] if k not in META]
    for r in rows:
        for m in metrics:
            r[m] = float(r[m])
        r["eps_255"] = int(float(r["eps_255"]))
    return rows, metrics


def deltas(rows, metrics):
    clean = {(r["arm"], r["seq"], r["frame"]): r for r in rows if r["attack"] == "clean"}
    out = defaultdict(dict)   # (arm, attack, eps, metric) -> {(seq, frame): delta}
    for r in rows:
        if r["attack"] == "clean":
            continue
        c = clean[(r["arm"], r["seq"], r["frame"])]
        for m in metrics:
            out[(r["arm"], r["attack"], r["eps_255"], m)][(r["seq"], r["frame"])] = r[m] - c[m]
    return out, clean


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    adj = np.full_like(p, np.nan)
    running, m = 0.0, int(np.sum(~np.isnan(p)))
    for rank, i in enumerate(order):
        if np.isnan(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def compare(d, ref, cmp_, attacks, eps_list, metrics):
    results = []
    for a in attacks:
        for e in eps_list:
            for m in metrics:
                dr, dc = d.get((ref, a, e, m)), d.get((cmp_, a, e, m))
                if not dr or not dc:
                    continue
                keys = sorted(set(dr) & set(dc))
                xr = np.array([dr[k] for k in keys])
                xc = np.array([dc[k] for k in keys])
                diff = xc - xr
                sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
                try:
                    p = wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan
                except ValueError:
                    p = np.nan
                lo, hi = boot_ci(diff) if len(diff) > 1 else (np.nan, np.nan)
                results.append(dict(attack=a, eps_255=e, metric=m, n=len(diff),
                                    mean_d_ref=xr.mean(), mean_d_cmp=xc.mean(),
                                    mean_diff=diff.mean(), ci_lo=lo, ci_hi=hi,
                                    d_z=diff.mean() / sd if sd and sd > 0 else np.nan,
                                    p=p))
    for r, pa in zip(results, holm([r["p"] for r in results])):
        r["p_holm"] = pa
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cmp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    start_log(ROOT, f"analyze_{args.cmp}_vs_{args.ref}")

    rows, metrics = load(args.csv)
    d, clean = deltas(rows, metrics)
    attacks = sorted({r["attack"] for r in rows if r["attack"] != "clean"})
    eps_list = sorted({r["eps_255"] for r in rows if r["attack"] != "clean"})

    # Clean-state differences first: if MNAT starts at a different density,
    # a delta comparison alone can mislead (ceiling / floor effects).
    print("Clean means (ref vs cmp):")
    for m in PRIMARY:
        vr = [c[m] for (arm, *_), c in clean.items() if arm == args.ref]
        vc = [c[m] for (arm, *_), c in clean.items() if arm == args.cmp]
        if vr and vc:
            print(f"  {m:18s} {np.mean(vr):10.4g}  {np.mean(vc):10.4g}")

    res = compare(d, args.ref, args.cmp, attacks, eps_list, metrics)

    out = args.out or args.csv.replace(".csv", f"_paired_{args.cmp}_vs_{args.ref}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)

    print(f"\nPrimary metrics  (diff = delta_{args.cmp} - delta_{args.ref}, paired over frames)")
    print(f"{'attack':14s} {'eps':>4s} {'metric':18s} {'d_ref':>10s} {'d_cmp':>10s} "
          f"{'diff':>10s} {'95% CI':>23s} {'d_z':>6s} {'p_holm':>8s}")
    for r in res:
        if r["metric"] not in PRIMARY:
            continue
        print(f"{r['attack']:14s} {r['eps_255']:>4d} {r['metric']:18s} "
              f"{r['mean_d_ref']:10.4g} {r['mean_d_cmp']:10.4g} {r['mean_diff']:10.4g} "
              f"[{r['ci_lo']:10.4g},{r['ci_hi']:10.4g}] {r['d_z']:6.2f} {r['p_holm']:8.2g}")
    print(f"\nfull table: {out}")
    print("Specificity check: a diff under sponge_l0 / det_inflation only means something "
          "if the same metric's diff under random and evasion is clearly smaller.")


if __name__ == "__main__":
    main()
