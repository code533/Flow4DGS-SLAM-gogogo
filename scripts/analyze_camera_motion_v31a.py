"""Paired V3.1a Shadow Mode evaluation on one run.

Ground truth is read only from saved diagnostics; never used by either solver.
"""
import argparse
import csv
import glob
import os
from collections import Counter
import numpy as np


def relative_metrics(Tprev, Tnext, GTprev, GTnext):
    A = np.linalg.solve(Tprev, Tnext)
    B = np.linalg.solve(GTprev, GTnext)
    te = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    x = A[:3, :3] @ B[:3, :3].T
    u, _, vh = np.linalg.svd(x)
    correction = np.diag([1., 1., np.linalg.det(u @ vh)])
    x = u @ correction @ vh
    rot = float(np.rad2deg(np.arccos(np.clip((np.trace(x) - 1) / 2, -1, 1))))
    am = float(np.linalg.norm(A[:3, 3]))
    bm = float(np.linalg.norm(B[:3, 3]))
    magnitude_error = abs(am - bm)
    direction_angle = float("nan")
    if am > 1e-6 and bm > 1e-6:
        cosine = np.dot(A[:3, 3], B[:3, 3]) / (am * bm)
        direction_angle = float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))
    return te, rot, magnitude_error, direction_angle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", help="Single run directory (not all experiments)")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()
    files = sorted(glob.glob(os.path.join(
        args.result_root, "**", "camera_motion_v31a_*.npz"), recursive=True))
    if not files:
        print("Files: 0. Check flow_cam_v31a.enabled/save_debug and Results.save_dir.")
        return
    statuses = Counter()
    rows = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            status = str(d["status"].item())
            statuses[status] += 1
            row = {"frame_id": int(d["frame_id"]), "status": status, "file": f}
            keys = ("n_common", "n_holdout", "legacy_common_median_px",
                    "geo_common_median_px", "legacy_holdout_rmse_px",
                    "geo_holdout_rmse_px", "legacy_holdout_median_px",
                    "geo_holdout_median_px", "condition")
            for key in keys:
                row[key] = float(d[key].item()) if key in d else float("nan")
            needed = ("T_previous_est", "T_candidate_legacy_common",
                      "T_candidate_geo", "T_previous_gt", "T_current_gt")
            if all(k in d for k in needed):
                for name, pose in (("legacy", d["T_candidate_legacy_common"]),
                                   ("geo", d["T_candidate_geo"])):
                    te, re, me, de = relative_metrics(
                        d["T_previous_est"], pose,
                        d["T_previous_gt"], d["T_current_gt"])
                    row[name + "_translation_error_m"] = te
                    row[name + "_rotation_error_deg"] = re
                    row[name + "_magnitude_error_m"] = me
                    row[name + "_direction_error_deg"] = de
                if "T_tracking_initial_actual" in d:
                    te, re, _, _ = relative_metrics(
                        d["T_previous_est"], d["T_tracking_initial_actual"],
                        d["T_previous_gt"], d["T_current_gt"])
                    row["actual_translation_error_m"] = te
                    row["actual_rotation_error_deg"] = re
            rows.append(row)
    print("Files:", len(files), "Statuses:", dict(statuses))
    for metric, description in (
        ("holdout_rmse_px", "Held-out flow RMSE (px)"),
        ("holdout_median_px", "Held-out median flow residual (px)"),
        ("translation_error_m", "Relative translation vector error (m)"),
        ("rotation_error_deg", "Relative rotation error (degrees)"),
        ("direction_error_deg", "Translation direction error (degrees)"),
        ("magnitude_error_m", "Translation magnitude error (m)"),
    ):
        a, b = [], []
        for row in rows:
            x, y = row.get("legacy_" + metric, float("nan")), row.get("geo_" + metric, float("nan"))
            if np.isfinite(x) and np.isfinite(y):
                a.append(x)
                b.append(y)
        if a:
            a, b = np.asarray(a), np.asarray(b)
            diff = b - a
            print(f"{description}: paired={len(a)}, legacy median={np.median(a):.6f}, "
                  f"geo median={np.median(b):.6f}, geo-legacy mean={np.mean(diff):+.6f}, "
                  f"geo better={(diff < 0).mean() * 100:.1f}%")
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        columns = sorted(set().union(*(r.keys() for r in rows)))
        with open(args.csv, "w", newline="", encoding="utf-8") as h:
            wr = csv.DictWriter(h, fieldnames=columns)
            wr.writeheader()
            wr.writerows(rows)
        print("CSV saved:", args.csv)


if __name__ == "__main__":
    main()
