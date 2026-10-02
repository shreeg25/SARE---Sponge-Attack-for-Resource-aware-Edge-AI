"""
Collect every replay result into one comparison (the paper's main table).

Reads results/pipeline/<model>/replay_<attack>_<guard>.json for all models and
writes results/compare/pipeline_comparison.csv (one row per model x attack x
guard x condition, plus an 'all' row pooling the four conditions). Pooled MOTA
and IDF1 are recomputed from the summed error counts, not averaged.

    python scripts/compare.py                 # prints the pooled table
    python scripts/compare.py --condition night
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import start_log

PER_FRAME = ["dets_in_per_frame", "reid_calls_per_frame", "tracks_out_per_frame", "uplink_B_per_frame",
             "tracker_ms_mean", "guard_ms_mean", "reid_ms_per_frame", "pipeline_ms_per_frame"]
COUNTS = ["num_false_positives", "num_misses", "num_switches", "num_objects", "idtp", "idfp", "idfn"]
ATTACK_ORDER = ["clean", "linf", "l2", "l1", "inflate", "sponge_act", "uap", "flood"]


def pool(conds):
    frames = sum(c["frames"] for c in conds)
    out = {"frames": frames}
    for k in PER_FRAME:
        if all(k in c for c in conds):
            out[k] = sum(c[k] * c["frames"] for c in conds) / frames
    out["reid_calls_p95"] = max(c["reid_calls_p95"] for c in conds)
    tot = {k: sum(c.get(k, 0) for c in conds) for k in COUNTS}
    out.update(tot)
    out["mota"] = 1 - (tot["num_false_positives"] + tot["num_misses"] + tot["num_switches"]) / max(tot["num_objects"], 1)
    out["idf1"] = 2 * tot["idtp"] / max(2 * tot["idtp"] + tot["idfp"] + tot["idfn"], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="all", help="all | day | night | rain | night_rain")
    args = ap.parse_args()
    start_log(ROOT, "compare")
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "pipeline" / "*" / "replay_*.json"))):
        d = json.load(open(f))
        model = Path(f).parent.name
        res = d["results"]
        for cond, r in res.items():
            rows.append({"model": model, "attack": d["attack"], "guard": d["guard"], "condition": cond, **r})
        if all("idtp" in r for r in res.values()):
            rows.append({"model": model, "attack": d["attack"], "guard": d["guard"], "condition": "all",
                         **pool(list(res.values()))})
    if not rows:
        sys.exit("no replay results found under results/pipeline/")
    out = ROOT / "results" / "compare" / "pipeline_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "attack", "guard", "condition"), k))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    sel = [r for r in rows if r["condition"] == args.condition]
    sel.sort(key=lambda r: (ATTACK_ORDER.index(r["attack"]) if r["attack"] in ATTACK_ORDER else 99,
                            r["model"], r["guard"]))
    print(f"\ncondition: {args.condition}   ({len(sel)} rows; full table in {out})")
    print(f"{'attack':10s} {'model':26s} {'guard':12s} {'det/f':>6s} {'reid/f':>6s} {'p95':>4s} {'up B/f':>7s} "
          f"{'pipe ms':>8s} {'MOTA':>7s} {'IDF1':>6s}")
    last = None
    for r in sel:
        if last is not None and r["attack"] != last:
            print()
        last = r["attack"]
        pm = r.get("pipeline_ms_per_frame", float("nan"))
        print(f"{r['attack']:10s} {r['model']:26s} {r['guard']:12s} {r['dets_in_per_frame']:6.2f} "
              f"{r['reid_calls_per_frame']:6.2f} {r['reid_calls_p95']:4.0f} {r['uplink_B_per_frame']:7.1f} "
              f"{pm:8.2f} {r['mota']:7.3f} {r['idf1']:6.3f}")


if __name__ == "__main__":
    main()
