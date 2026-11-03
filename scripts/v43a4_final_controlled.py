"""V4.3a.4 FINAL: locked, one-shot cross-sequence source-score validation.

Run once on previously unseen rgbd_bonn_balloon outputs. The source
balloon2 CSV may be supplied for CONTEXT ONLY, never calibration.
Primary switch margin=0; secondary sensitivity=0.2mm (already defined
before the holdout run). Depth evaluation is FUTURE RAW MAE for all
policies, with candidate-independent future dynamic depth mask.
Read-only; no online propagation modification.

No threshold search or 'best on validation' ranking permitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

METHODS=("g0","g3_support","g4")
POLICIES=("always_g0","mae","clipped50","huber20","interior_clipped50")
PRIMARY_MARGIN_M=0.0
SECONDARY_MARGIN_M=0.0002
FUTURE_TIE_TOL_M=0.0001
SOURCE_COVERAGE_TOL=0.005
MIN_DEPTH_COVERAGE=0.95
MIN_DYNAMIC_PIXELS=256
MIN_PAIRS=12
# One-time go/no-go gate, frozen BEFORE the independent-scene results:
MEAN_ROBUST_ADVANTAGE_M=.0005
ORACLE_ADVANTAGE_FRACTION=.05
MAX_RGB_WORSEN=.0005
MAX_COVERAGE_DROP=.005
MIN_DISCORDANT=5


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resolved_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model=cfg["model_params"]
    obs=model["v43a_observation"]
    scene=cfg["Dataset"]["dataset_path"].rstrip("/")
    if not scene.endswith("/rgbd_bonn_balloon"):
        raise ValueError("Heldout scene must be Bonn balloon, NOT balloon2: "+scene)
    checks={
        "G0 online mapping":model["v42a_propagation"]["mode"]=="g0",
        "offset dynamic Gaussian":model["dynamic_model"]=="offset",
        "flow_offset":bool(model["flow_offset"]),
        "use_knn":bool(model["use_knn"]),
        "full-resolution flow":cfg["Training"]["flow_ds"]==1,
        "read-only comparisons":bool(obs["enabled"]),
        "frozen temporal enabled":bool(obs["temporal_enabled"]),
        "prediction enabled":bool(obs["predictive_enabled"]),
        "predictive margin ZERO":float(obs["predictive_decision_margin_m"])==0,
        "sampling interval FOUR":int(obs["predictive_source_every_n_event"])==4,
        "source cap 25":int(obs["predictive_max_sources"])==25,
        "max next-frame gap 12":int(obs["temporal_max_frame_gap"])==12,
        "min source dynamic depth pixels":int(obs["predictive_min_dynamic_depth_pixels"])==MIN_DYNAMIC_PIXELS,
        "min depth coverage":float(obs["predictive_min_coverage_fraction"])==MIN_DEPTH_COVERAGE,
        "depth clipped at 50mm":float(obs["predictive_clip_m"])==.05,
        "Huber delta 20mm":float(obs["predictive_huber_delta_m"])==.02,
        "mask erosion 5px":int(obs["predictive_interior_erosion_px"])==5,
        "no old failure/ablation":not any(bool(model["v42a_propagation"].get(k,False)) for k in
                                       ("attribute_failures","g34_shadow","g4_ablation")),
    }
    problems=[key for key,ok in checks.items() if not ok]
    if problems:
        raise ValueError("Heldout config does not match locked protocol: "+str(problems))
    return scene


def read_scores(path):
    df=pd.read_csv(path)
    required={"source_keyframe_id","future_keyframe_id","timepoint","hypothesis",
              "mae","clipped50","huber20","interior_clipped50","rgb_mae",
              "depth_coverage","n_dynamic_depth","n_interior_depth","n_dynamic_rgb"}
    if not required.issubset(df.columns):
        raise ValueError("Missing score CSV columns: "+str(sorted(required-set(df.columns))))
    keys=["source_keyframe_id","future_keyframe_id","timepoint","hypothesis"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate candidate records")
    if set(df.timepoint)!={"source","future"}:
        raise ValueError("Need source AND future observations")
    if set(df.hypothesis)!=set(METHODS):
        raise ValueError("Incomplete three-method comparison")
    numeric=["mae","clipped50","huber20","rgb_mae","depth_coverage"]
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite required candidate scores")
    if (df[numeric].to_numpy(dtype=float)<0).any():
        raise ValueError("Negative candidate metric")
    if ((df.depth_coverage>1)|(df.depth_coverage<0)).any():
        raise ValueError("Coverage out of range")
    groups=df.groupby(["source_keyframe_id","future_keyframe_id"],sort=True)
    grouped=[]
    for (src,fut),data in groups:
        if int(fut)<=int(src):
            raise ValueError("Future must follow source")
        if len(data)!=6 or data.groupby("timepoint").hypothesis.apply(set).to_dict()!={
                    "source":set(METHODS),"future":set(METHODS)}:
            raise ValueError(f"Unpaired candidate data: {src}->{fut}")
        by={}
        for timepoint in ("source","future"):
            t=data[data.timepoint==timepoint].set_index("hypothesis")
            if t.n_dynamic_depth.nunique()!=1 or t.n_dynamic_rgb.nunique()!=1:
                raise ValueError("Within-frame GT pixel counts differ across methods")
            if int(t.n_dynamic_depth.iloc[0])<MIN_DYNAMIC_PIXELS:
                raise ValueError(f"Insufficient dynamic depth pixels: {src}->{fut}")
            if float(t.depth_coverage.min())<MIN_DEPTH_COVERAGE:
                raise ValueError(f"Depth coverage below preregistered floor: {src}->{fut}")
            by[timepoint]={name:t.loc[name].to_dict() for name in METHODS}
        grouped.append((int(src),int(fut),by))
    # Ensure only the target source timepoint and not future is used to
    # make source choices; caller never passes future data to select().
    return grouped


def select(source,policy,margin):
    if policy=="always_g0":
        return "g0"
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    base=float(source["g0"][policy])
    if not np.isfinite(base):
        return "g0"
    eligible=[]
    for method in METHODS[1:]:
        val=float(source[method][policy])
        if np.isfinite(val) and (
            float(source[method]["depth_coverage"]) >=
            float(source["g0"]["depth_coverage"])-SOURCE_COVERAGE_TOL):
            eligible.append(method)
    if not eligible:
        return "g0"
    best=min(eligible,key=lambda m:(source[m][policy],METHODS.index(m)))
    return best if source[best][policy]<base-margin else "g0"


def evaluate_pairs(groups,margin,split):
    rows=[]
    for src,fut,d in groups:
        s=d["source"]
        f=d["future"]
        oracle=min(float(f[m]["mae"]) for m in METHODS)
        for policy in POLICIES:
            selected=select(s,policy,margin)
            selected_future=float(f[selected]["mae"])
            g0_future=float(f["g0"]["mae"])
            rows.append(dict(
                split=split,source_keyframe_id=src,future_keyframe_id=fut,
                frame_gap=fut-src,policy=policy,chosen=selected,
                margin_m=margin,
                future_gain_vs_g0_m=g0_future-selected_future,
                future_regret_m=selected_future-oracle,
                future_near_oracle=selected_future<=oracle+FUTURE_TIE_TOL_M,
                future_rgb_change=float(f[selected]["rgb_mae"])-float(f["g0"]["rgb_mae"]),
                future_coverage_change=(float(f[selected]["depth_coverage"])-
                                        float(f["g0"]["depth_coverage"])),
                source_score=float(s[selected].get(policy,float("nan"))),
                source_g0_score=float(s["g0"].get(policy,float("nan"))),
                future_oracle_depth_mae_m=oracle,
                future_g0_depth_mae_m=g0_future,
            ))
    return pd.DataFrame(rows)


def aggregate(per_pair):
    result=[]
    for (split,margin,policy),g in per_pair.groupby(["split","margin_m","policy"],sort=False):
        result.append(dict(
            split=split,margin_m=margin,policy=policy,n_pairs=len(g),
            choose_non_g0_n=int((g.chosen!="g0").sum()),
            mean_future_gain_mm=float(g.future_gain_vs_g0_m.mean()*1000),
            median_future_gain_mm=float(g.future_gain_vs_g0_m.median()*1000),
            mean_regret_mm=float(g.future_regret_m.mean()*1000),
            near_oracle_fraction=float(g.future_near_oracle.mean()),
            mean_future_rgb_change=float(g.future_rgb_change.mean()),
            mean_future_coverage_change=float(g.future_coverage_change.mean())))
    return pd.DataFrame(result)


def contrast(per_pair, margin=PRIMARY_MARGIN_M,split="heldout"):
    target=per_pair[(per_pair.split==split)&(per_pair.margin_m==margin)]
    table=target.pivot(index=["source_keyframe_id","future_keyframe_id"],
                       columns="policy",values="future_gain_vs_g0_m")
    choice=target.pivot(index=["source_keyframe_id","future_keyframe_id"],
                        columns="policy",values="chosen")
    g=table["clipped50"]-table["mae"]
    d=choice["clipped50"]!=choice["mae"]
    rgb=target.pivot(index=["source_keyframe_id","future_keyframe_id"],
                     columns="policy",values="future_rgb_change")
    cov=target.pivot(index=["source_keyframe_id","future_keyframe_id"],
                     columns="policy",values="future_coverage_change")
    oracle=target.pivot(index=["source_keyframe_id","future_keyframe_id"],
                        columns="policy",values="future_near_oracle")
    contrasts=pd.DataFrame({
        "source_keyframe_id":[int(x[0]) for x in table.index],
        "future_keyframe_id":[int(x[1]) for x in table.index],
        "mae_choice":choice["mae"].to_numpy(),
        "clipped50_choice":choice["clipped50"].to_numpy(),
        "discordant":d.to_numpy(),
        "clipped_minus_mae_future_gain_mm":g.to_numpy()*1000,
        "mae_future_gain_mm":table["mae"].to_numpy()*1000,
        "clipped50_future_gain_mm":table["clipped50"].to_numpy()*1000,
    })
    stats=dict(
        n_pairs=len(table),
        discordant_decisions=int(d.sum()),
        clipped_advantage_mean_mm=float(g.mean()*1000),
        clipped_advantage_median_mm=float(g.median()*1000),
        clipped_better_decisions=int((g>FUTURE_TIE_TOL_M).sum()),
        mae_better_decisions=int((g< -FUTURE_TIE_TOL_M).sum()),
        clipped_minus_mae_oracle_fraction=float(oracle["clipped50"].mean()-oracle["mae"].mean()),
        clipped_minus_mae_rgb_mean=float((rgb["clipped50"]-rgb["mae"]).mean()),
        clipped_minus_mae_coverage_mean=float((cov["clipped50"]-cov["mae"]).mean()),
        clipped_mean_gain_vs_g0_mm=float(table["clipped50"].mean()*1000),
        mae_mean_gain_vs_g0_mm=float(table["mae"].mean()*1000)
    )
    return contrasts,stats


def decision(stats):
    if stats["n_pairs"]<MIN_PAIRS:
        return "INSUFFICIENT_DATA_NO_GO", [
            "Less than 12 complete frozen source/next-keyframe pairs"]
    failures=[]
    if stats["discordant_decisions"]<MIN_DISCORDANT:
        failures.append("Fewer than 5 differing MAE-vs-Clipped50 choices")
    if stats["clipped_advantage_mean_mm"]<MEAN_ROBUST_ADVANTAGE_M*1000:
        failures.append("Clipped50 future mean gain advantage <0.5mm")
    if stats["clipped_advantage_median_mm"] < -FUTURE_TIE_TOL_M*1000:
        failures.append("Clipped50 future paired median substantially worse")
    if stats["clipped_minus_mae_oracle_fraction"]<ORACLE_ADVANTAGE_FRACTION:
        failures.append("Clipped50 near-oracle fraction gain <5pp")
    if stats["clipped_minus_mae_rgb_mean"]>MAX_RGB_WORSEN:
        failures.append("Clipped50 future RGB regression >0.0005 vs MAE")
    if stats["clipped_minus_mae_coverage_mean"]< -MAX_COVERAGE_DROP:
        failures.append("Clipped50 depth coverage drop >0.5pp vs MAE")
    if stats["clipped_mean_gain_vs_g0_mm"]<=0:
        failures.append("Clipped50 not better than always-G0")
    if failures:
        return "NO_GO_STOP_SCORE_SELECTION",failures
    return "PASS_ONLY_TO_ONLINE_ABLATION",[
        "Clipped50 passes this one-scene screening gate; still requires "
        "independent repeated runs and actual online Mapping ablation"]


def run_control(holdout_csv,holdout_config,output,development_csv=None):
    holdout_csv=Path(holdout_csv)
    holdout_config=Path(holdout_config)
    output=Path(output)
    scene=validate_resolved_config(holdout_config)
    heldout=read_scores(holdout_csv)
    samples=[]
    for margin in (PRIMARY_MARGIN_M,SECONDARY_MARGIN_M):
        samples.append(evaluate_pairs(heldout,margin,"heldout"))
    if development_csv is not None:
        development_csv=Path(development_csv)
        train=read_scores(development_csv)
        for margin in (PRIMARY_MARGIN_M,SECONDARY_MARGIN_M):
            samples.append(evaluate_pairs(train,margin,"development"))
    all_rows=pd.concat(samples,ignore_index=True)
    summary=aggregate(all_rows)
    contrast_rows,stats=contrast(all_rows)
    result,reasons=decision(stats)
    output.mkdir(parents=True,exist_ok=True)
    all_rows.to_csv(output/"v43a4_final_policy_per_pair.csv",index=False)
    summary.to_csv(output/"v43a4_final_policy_summary.csv",index=False)
    contrast_rows.to_csv(output/"v43a4_final_mae_vs_clipped50.csv",index=False)
    manifest=dict(
        protocol="V4.3a.4 final locked cross-sequence comparison",
        primary_margin_m=PRIMARY_MARGIN_M,
        secondary_margin_m=SECONDARY_MARGIN_M,
        validation_scene=scene,
        validation_scores_sha256=digest(holdout_csv),
        validation_config_sha256=digest(holdout_config),
        development_scores_sha256=digest(development_csv)
                     if development_csv else None,
        source_sampling="first eligible propagation event, then every 4th",
        source_keyframe_cap=25,
        future_metric="ordinary dynamic depth MAE",
        frozen_motion="zero-order hold at next keyframe",
        primary_comparison="Clipped50 versus MAE (source-only score)",
        decision=result,reasons=reasons,
        primary_contrast=stats,
    )
    (output/"v43a4_final_protocol_and_decision.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=[
        "# Final controlled candidate-scoring decision",
        "",
        "**Decision: "+result+"**",
        "",
        "Evaluation: independent Bonn balloon sequence, one frozen-map "
        "zero-order-hold short-term test.",
        "All policies use the same future ordinary depth MAE; source-only selection.",
        "",
        f"Complete pairs: {stats['n_pairs']}; discordant policy decisions: {stats['discordant_decisions']}.",
        f"Clipped50 minus MAE mean future gain: {stats['clipped_advantage_mean_mm']:+.4f} mm.",
        f"Clipped50 minus MAE median pair gain: {stats['clipped_advantage_median_mm']:+.4f} mm.",
        f"Clipped50 minus MAE near-oracle rate: {100*stats['clipped_minus_mae_oracle_fraction']:+.2f} pp.",
        "",
        "Predeclared gate reasons:"
    ]+["- "+r for r in reasons]+[
        "",
        "Secondary 0.2mm margin is reported for sensitivity only; "
        "it must not override this zero-margin primary decision.",
        "Single scene and correlated keyframes: descriptive screening only, "
        "not proof of generalization, ATE improvement or correct 3D motion."
    ]
    (output/"v43a4_final_decision.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--holdout-scores",required=True)
    ap.add_argument("--holdout-config",required=True,
                    help="SLAM run directory's resolved config.yml")
    ap.add_argument("--development-scores",default=None)
    ap.add_argument("--output",required=True)
    a=ap.parse_args()
    report=run_control(a.holdout_scores,a.holdout_config,a.output,a.development_scores)
    print("FINAL:",report["decision"])
    print("PAIRS:",report["primary_contrast"]["n_pairs"])
    print("ROBUST_MINUS_MAE_GAIN_MM:",
          round(report["primary_contrast"]["clipped_advantage_mean_mm"],6))
    for reason in report["reasons"]:print(" -",reason)
    print("Files in:",a.output)


if __name__=="__main__":
    main()
