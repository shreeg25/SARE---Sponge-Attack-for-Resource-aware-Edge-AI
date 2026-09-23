"""
Build SARE's 2D detection index from nuScenes CAM_FRONT keyframes. Run once,
in an environment with nuscenes-devkit installed (training does not need it):

    python scripts/prepare_nuscenes.py --root data/nuscenes --version v1.0-mini
    python scripts/prepare_nuscenes.py --root data/nuscenes --version v1.0-trainval

Writes data/nuscenes_2d.json (same name for both versions, so configs never
change) and prints scene/frame/box counts per split and condition.

2D boxes use the devkit's own reprojection: 3D box -> camera frame, drop
corners behind the camera, project, intersect the hull with the image.

Conditions come from the human-written scene descriptions:
    night -> "night" in description; rain -> "rain" in description (includes
    "after rain", i.e. wet roads); night_rain -> both; day -> neither.

Splits:
    train   day scenes from the official train split, minus the select subset
    select  every 5th day train scene (by name) -- model selection only
    test    official val scenes (all conditions) + night/rain train scenes,
            which training never sees
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {1: "pedestrian", 2: "vehicle", 3: "cyclist"}
PED_KEEP = {"adult", "child", "construction_worker", "police_officer"}
VIS_KEEP = {"2", "3", "4"}          # >= 40% visible across the camera rig
SELECT_EVERY = 5
CAMERA = "CAM_FRONT"


def map_category(name):
    if name.startswith("human.pedestrian.") and name.split(".")[-1] in PED_KEEP:
        return 1
    if name in ("vehicle.bicycle", "vehicle.motorcycle"):
        return 3
    if name.startswith("vehicle."):     # car, truck, bus.*, emergency.*, construction, trailer
        return 2
    return None


def boxes_for(nusc, sd_token):
    sd = nusc.get("sample_data", sd_token)
    sample = nusc.get("sample", sd["sample_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    K = np.array(cs["camera_intrinsic"])
    imsize = (sd["width"], sd["height"])

    boxes, labels, ids = [], [], []
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        lab = map_category(ann["category_name"])
        if lab is None or ann["visibility_token"] not in VIS_KEEP:
            continue
        box = nusc.get_box(tok)
        box.translate(-np.array(pose["translation"]))
        box.rotate(Quaternion(pose["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(Quaternion(cs["rotation"]).inverse)
        corners = box.corners()
        corners = corners[:, corners[2, :] > 0]
        if corners.shape[1] == 0:
            continue
        coords = view_points(corners, K, True).T[:, :2].tolist()
        fc = post_process_coords(coords, imsize)
        if fc is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in fc)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
        labels.append(lab)
        ids.append(ann["instance_token"])
    return sd["filename"], boxes, labels, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="data/nuscenes_2d.json")
    ap.add_argument("--keep-missing", action="store_true",
                    help="index frames whose image file is absent (default: skip them, "
                         "so a partial blob download still produces a usable index)")
    args = ap.parse_args()

    root = Path(args.root) if Path(args.root).is_absolute() else ROOT / args.root
    nusc = NuScenes(version=args.version, dataroot=str(root), verbose=True)
    sp = create_splits_scenes()
    mini = "mini" in args.version
    train_names = set(sp["mini_train" if mini else "train"])
    val_names = set(sp["mini_val" if mini else "val"])

    scenes = []
    for sc in nusc.scene:
        d = sc["description"].lower()
        night, rain = "night" in d, "rain" in d
        cond = "night_rain" if night and rain else "night" if night else "rain" if rain else "day"
        if sc["name"] in train_names:
            official = "train"
        elif sc["name"] in val_names:
            official = "val"
        else:
            continue
        scenes.append((sc, cond, official, night, rain))

    day_train = sorted(sc["name"] for sc, c, o, *_ in scenes if o == "train" and c == "day")
    select_names = set(day_train[::SELECT_EVERY])

    records = []
    missing = 0
    stats = defaultdict(lambda: {"scenes": 0, "frames": 0, **{v: 0 for v in CLASSES.values()}})
    for sc, cond, official, night, rain in scenes:
        if official == "val" or cond != "day":
            split = "test"
        elif sc["name"] in select_names:
            split = "select"
        else:
            split = "train"
        st = stats[(split, cond)]
        scene_frames = 0
        tok = sc["first_sample_token"]
        while tok:
            s = nusc.get("sample", tok)
            fname, boxes, labels, ids = boxes_for(nusc, s["data"][CAMERA])
            tok = s["next"]
            if not args.keep_missing and not (root / fname).is_file():
                missing += 1
                continue
            records.append({"file": fname, "scene": sc["name"], "split": split, "condition": cond,
                            "is_night": night, "is_rain": rain,
                            "timestamp": s["timestamp"], "boxes": boxes, "labels": labels, "ids": ids})
            scene_frames += 1
            st["frames"] += 1
            for l in labels:
                st[CLASSES[l]] += 1
        if scene_frames:
            st["scenes"] += 1

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"version": args.version, "camera": CAMERA, "classes": CLASSES,
                   "records": records}, fh)

    print(f"\nwrote {len(records)} frames to {out}"
          + (f"  ({missing} keyframes skipped: image file not downloaded)" if missing else "") + "\n")
    print(f"{'split':7s} {'cond':6s} {'scenes':>6s} {'frames':>7s} {'ped':>7s} {'vehicle':>8s} {'cyclist':>8s}")
    for (split, cond), st in sorted(stats.items()):
        print(f"{split:7s} {cond:6s} {st['scenes']:6d} {st['frames']:7d} {st['pedestrian']:7d} "
              f"{st['vehicle']:8d} {st['cyclist']:8d}")


if __name__ == "__main__":
    main()
