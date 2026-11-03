"""Summarize V5.1a strict FOV projection and update ablation.

The keyframe CSV is generated before GaussianModel.update_delta_xyz().
Shadow measures exclusion and KNN recomputation without changing live G0.
Online ATE/PSNR/SSIM need separate standard SLAM evaluation.
"""
import argparse
from pathlib import Path
import pandas as pd


def summarize(folder):
    folder=Path(folder)
    data=pd.read_csv(folder/"v51a_per_keyframe.csv")
    if data.empty:
        raise ValueError("Empty V5.1a diagnostics")
    if data.keyframe_id.duplicated().any():
        raise ValueError("Repeated keyframe IDs in one run")
    if data["mode"].nunique()!=1:
        raise ValueError("Mixed modes within one run; compare separate directories")
    if (data.n_removed_candidates != data.n_legacy_candidates-data.n_strict_candidates).any():
        raise ValueError("Candidate accounting mismatch")
    if (data.n_removed_source_fov+data.n_removed_target_fov-
        data.n_removed_both_fov != data.n_removed_candidates).any():
        raise ValueError("FOV exclusion categories do not form a union")
    out={
        "mode":str(data["mode"].iloc[0]),
        "n_keyframes":len(data),
        "all_dynamic_records":int(data.n_all_dynamic.sum()),
        "legacy_candidate_records":int(data.n_legacy_candidates.sum()),
        "strict_candidate_records":int(data.n_strict_candidates.sum()),
        "removed_candidate_records":int(data.n_removed_candidates.sum()),
        "removed_source_records":int(data.n_removed_source_fov.sum()),
        "removed_target_records":int(data.n_removed_target_fov.sum()),
        "removed_both_records":int(data.n_removed_both_fov.sum()),
        "removed_legacy_gate_accepted_records":int(data.n_removed_that_pass_legacy_0p3m_gate.sum()),
        "legacy_gate_accepted_records":int(data.n_legacy_gate_pass.sum()),
        "common_gate_flips":int(data.n_common_gate_changed.clip(lower=0).sum()),
        "mean_shared_strict_vs_legacy_m":
            float(data.shared_strict_vs_legacy_g0_mean_m.mean()),
        "mean_per_kf_removed_fraction":float(data.fraction_candidates_rejected.mean()),
    }
    out["removed_fraction_weighted"]= (
        out["removed_candidate_records"]/out["legacy_candidate_records"]
        if out["legacy_candidate_records"] else float("nan"))
    out["legacy_gate_accepted_removed_fraction"]=(
        out["removed_legacy_gate_accepted_records"]/out["legacy_gate_accepted_records"]
        if out["legacy_gate_accepted_records"] else float("nan"))
    summary=pd.DataFrame([out])
    summary.to_csv(folder/"v51a_summary.csv",index=False)
    lines=["V5.1a FOV validity diagnostic",f"Mode: {out['mode']}",
           f"Eligible frames: {out['n_keyframes']}",
           f"Legacy candidate records: {out['legacy_candidate_records']}",
           f"Strict candidate records: {out['strict_candidate_records']}",
           f"FOV-removed records: {out['removed_candidate_records']} "
           f"({out['removed_fraction_weighted']:.2%})",
           f"Removed records that legacy would accept under 0.3m: "
           f"{out['removed_legacy_gate_accepted_records']}",
           f"Legacy accepted motion removed: "
           f"{out['legacy_gate_accepted_removed_fraction']:.2%}",
           f"Mean shared strict-vs-legacy smooth displacement: "
           f"{out['mean_shared_strict_vs_legacy_m']:.5f} m",
           f"Shared gate flips: {out['common_gate_flips']}",
           "Source/target counts indicate image-FOV consistency under the "
           "legacy source-NDC-plus-flow approximation, not true target-camera "
           "visibility/occlusion ground truth.",
           "Do not call this an ATE or reconstruction improvement." ]
    (folder/"v51a_summary.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    x=ap.parse_args()
    summarize(x.input)


if __name__=="__main__":
    main()
