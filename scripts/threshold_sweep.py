"""
Threshold sensitivity of the SARE guard (thr<K>_t<tau>) for the paper.

Reads results/compare/pipeline_comparison.csv (written by scripts/compare.py)
and, for every model x attack, reports cost and tracking quality against the
guard threshold tau, with no-guard and the fixed cap as references:

  results/sweep/sweep_table.csv          long table: model, attack, condition, guard, tau, metrics
  results/sweep/sweep_<attack>.png/.pdf  MOTA, IDF1 and ReID calls vs tau, one line per model,
                                         dashed = fixed cap (same K), dotted = no guard
  results/sweep/sweep_inflate.tex        booktabs table for the paper (MOTA / ReID per tau)

Choosing the headline tau WITHOUT test data. If select-split results exist
(models named *_select, from extract_dets.py --split select), the rule fixed
before looking at them is applied there:
    pick the tau with the highest MOTA under 'inflate', among taus whose clean
    IDF1 is within 0.01 of the fixed cap's clean IDF1.
The choice goes to results/sweep/chosen_tau.json and is marked in the figures
and the LaTeX table; test results are never used to choose it.

    python scripts/compare.py                  # refresh the CSV first
    python scripts/threshold_sweep.py
    python scripts/threshold_sweep.py --condition night
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.logutil import start_log

ATTACKS = ["clean", "inflate", "inflate_hi", "uap", "linf", "l2", "l1", "sponge_act"]
ATTACK_LABEL = {"clean": "Clean", "inflate": "Inflation sponge", "inflate_hi": "Adaptive inflation (tau=0.8)",
                "uap": "Universal sponge (8/255)", "linf": "PGD L-inf", "l2": "PGD L2", "l1": "PGD L1",
                "sponge_act": "Activation sponge"}
MODEL_LABEL = {"nuscenes_baseline_s0": "Baseline", "nuscenes_bgat_s0": "BG-AT",
               "nuscenes_linf_at_s0": "L-inf AT", "nuscenes_linf_at_aug_s0": "L-inf AT + aug"}
METRICS = [("mota", "MOTA"), ("idf1", "IDF1"), ("reid_calls_per_frame", "ReID calls / frame")]
THR = re.compile(r"^thr(\d+)_t(\d+)$")


def tau_of(guard):
    m = THR.match(guard)
    return int(m.group(2)) / 100 if m else None


def k_of(guard):
    m = THR.match(guard)
    return int(m.group(1)) if m else None


def choose_tau(df):
    """Selection rule on the select split. Returns (tau, details) or (None, reason)."""
    sel = df[df.model.str.endswith("_select") & (df.condition == "all")]
    if sel.empty:
        return None, "no select-split results (run extract_dets.py --split select, replay, compare)"
    out = {}
    for model, g in sel.groupby("model"):
        cap_clean = g[(g.attack == "clean") & (g.guard.str.startswith("cap"))]
        if cap_clean.empty:
            continue
        cap_idf1 = float(cap_clean.idf1.iloc[0])
        thr = g[g.tau.notna()]
        ok = thr[(thr.attack == "clean") & (thr.idf1 >= cap_idf1 - 0.01)].tau.unique()
        inf = thr[(thr.attack == "inflate") & thr.tau.isin(ok)]
        if inf.empty:
            continue
        best = inf.loc[inf.mota.idxmax()]
        out[model] = {"tau": float(best.tau), "inflate_mota": float(best.mota), "cap_clean_idf1": cap_idf1,
                      "eligible_taus": sorted(float(t) for t in ok)}
    if not out:
        return None, "select results lack clean+inflate rows for both cap and thr guards"
    taus = [v["tau"] for v in out.values()]
    return max(set(taus), key=taus.count), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/compare/pipeline_comparison.csv")
    ap.add_argument("--condition", default="all")
    ap.add_argument("--min-taus", type=int, default=4)
    args = ap.parse_args()
    start_log(ROOT, "threshold_sweep")

    path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    df = pd.read_csv(path)
    df["tau"] = df.guard.map(tau_of)
    out = ROOT / "results" / "sweep"
    out.mkdir(parents=True, exist_ok=True)

    chosen, info = choose_tau(df)
    if chosen is not None:
        json.dump({"tau": chosen, "rule": "max inflate MOTA s.t. clean IDF1 >= cap - 0.01, select split",
                   "per_model": info}, open(out / "chosen_tau.json", "w"), indent=2)
        print(f"headline tau chosen on select split: {chosen}   {info}")
    else:
        print(f"headline tau: not chosen ({info})")

    test = df[~df.model.str.endswith("_select") & (df.condition == args.condition)]
    taus = sorted(test.tau.dropna().unique())
    if len(taus) < args.min_taus:
        sys.exit(f"only {len(taus)} thresholds found ({taus}); replay with at least {args.min_taus}, "
                 f"e.g. --guards thr10_t50 thr10_t60 thr10_t70 thr10_t80")
    ks = sorted({k_of(g) for g in test.guard if THR.match(g)})
    cap = f"cap{ks[0]}"
    models = [m for m in MODEL_LABEL if m in set(test.model)] + sorted(set(test.model) - set(MODEL_LABEL))
    attacks = [a for a in ATTACKS if a in set(test.attack)]

    keep = test[test.tau.notna() | test.guard.isin(["none", cap])]
    cols = ["model", "attack", "condition", "guard", "tau", "mota", "idf1", "reid_calls_per_frame",
            "reid_calls_p95", "uplink_B_per_frame", "pipeline_ms_per_frame"]
    keep[[c for c in cols if c in keep.columns]].sort_values(["attack", "model", "tau"]).to_csv(
        out / "sweep_table.csv", index=False)

    print(f"\ncondition: {args.condition}   thresholds: {taus}   reference cap: {cap}")
    for attack in attacks:
        a = test[test.attack == attack]
        print(f"\n== {ATTACK_LABEL.get(attack, attack)}")
        head = f"{'model':16s} {'metric':8s} {'none':>8s} {cap:>8s} " + " ".join(f"{'t=' + str(t):>8s}" for t in taus)
        print(head)
        for m in models:
            g = a[a.model == m]
            if g.empty:
                continue
            for key, short in [("mota", "MOTA"), ("idf1", "IDF1"), ("reid_calls_per_frame", "ReID/f")]:
                ref = lambda name: g[g.guard == name][key]
                vals = [ref("none"), ref(cap)] + [g[g.tau == t][key] for t in taus]
                txt = " ".join(f"{v.iloc[0]:8.3f}" if len(v) else f"{'-':>8s}" for v in vals)
                print(f"{MODEL_LABEL.get(m, m):16s} {short:8s} {txt}")

        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
        for ax, (key, lab) in zip(axes, METRICS):
            # y-range from the guard lines and the cap only: no-guard values (e.g. MOTA -2.8) would
            # flatten everything; they are drawn only when they fall inside that range (and are in the table)
            vis = a[(a.tau.notna()) | (a.guard == cap)][key]
            lo, hi = float(vis.min()), float(vis.max())
            pad = 0.08 * max(hi - lo, 1e-3)
            for i, m in enumerate(models):
                g = a[a.model == m]
                line = g[g.tau.notna()].sort_values("tau")
                if line.empty:
                    continue
                color = f"C{i}"
                ax.plot(line.tau, line[key], "o-", color=color, ms=3, lw=1.2, label=MODEL_LABEL.get(m, m))
                c = g[g.guard == cap][key]
                if len(c):
                    ax.axhline(float(c.iloc[0]), color=color, ls="--", lw=0.8)
                n = g[g.guard == "none"][key]
                if len(n) and lo - pad <= float(n.iloc[0]) <= hi + pad:
                    ax.axhline(float(n.iloc[0]), color=color, ls=":", lw=0.8)
            if chosen is not None:
                ax.axvline(chosen, color="0.4", lw=0.8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel(r"guard threshold $\tau$")
            ax.set_ylabel(lab)
            ax.set_xticks(taus)
            ax.grid(alpha=0.3)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7, frameon=False)
        fig.suptitle(f"{ATTACK_LABEL.get(attack, attack)} ({args.condition}); dashed = {cap}, "
                     f"dotted = no guard (when in range; full values in the table)",
                     fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        stem = f"sweep_{attack}" + ("" if args.condition == "all" else f"_{args.condition}")
        fig.savefig(out / f"{stem}.png", dpi=200)
        fig.savefig(out / f"{stem}.pdf")
        plt.close(fig)

    if "inflate" in attacks:
        a = test[test.attack == "inflate"]
        hdr = ["No guard", f"Cap ({ks[0]})"] + [(r"$\tau$=" + f"{t:.1f}" + ("$^\\dagger$" if chosen == t else ""))
                                                 for t in taus]
        lines = [r"\begin{tabular}{l" + "c" * len(hdr) + "}", r"\toprule",
                 "Detector & " + " & ".join(hdr) + r" \\", r"\midrule"]
        for m in models:
            g = a[a.model == m]
            if g.empty:
                continue
            cells = []
            for sub in [g[g.guard == "none"], g[g.guard == cap]] + [g[g.tau == t] for t in taus]:
                cells.append(f"{sub.mota.iloc[0]:.3f} / {sub.reid_calls_per_frame.iloc[0]:.1f}" if len(sub) else "--")
            lines.append(f"{MODEL_LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}",
                  r"% cells: MOTA / ReID calls per frame under the inflation sponge, pooled over day, night, rain, night+rain."]
        if chosen is not None:
            lines.append(r"% $^\dagger$ headline threshold, chosen on the select split (never on test).")
        (out / "sweep_inflate.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote tables and figures to {out}")


if __name__ == "__main__":
    main()
