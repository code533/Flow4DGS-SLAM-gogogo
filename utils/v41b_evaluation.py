"""V4.1b: exact Unknown attribution and independent-label evaluation.

No SLAM modification. Input: V4.0b NPZ + V4.1a Shadow inference.
The original Flow4DGS mask is a comparator, NEVER a ground-truth label.
"""
from dataclasses import dataclass
import numpy as np

REASONS = {
    0: "DECIDED",
    1: "INVALID_FLOW",
    2: "LOW_TOTAL_EVIDENCE",
    3: "LOW_FLOW_AND_MAP_RELIABILITY",
    4: "FLOW_MAP_CONFLICT",
    5: "AMBIGUOUS_SCORE_BAND",
}

STATE_KEYS = (
    "legacy_binary",
    "state_flow_only",
    "state_naive_rgbd",
    "state_reliability_binary",
    "state_decoupled_3state",
)


def attribute_unknown(raw, shadow, cfg):
    """Partition *exactly* A3 Unknown pixels using V4.1a rule precedence.

    Reasons are mutually exclusive, exhaust all Unknown pixels. Secondary
    source cues (map uncovered, boundaries, possible occlusion) may overlap.
    """
    state = np.asarray(shadow["state_decoupled_3state"])
    flow_valid = np.asarray(shadow["flow_valid"], bool)
    evidence = np.asarray(shadow["evidence"], float)
    qf = np.asarray(shadow["q_flow"], float)
    qm = np.asarray(shadow["q_map"], float)
    motion = np.asarray(shadow["motion_score"], float)
    map_score = np.asarray(shadow["map_score"], float)
    p = np.asarray(shadow["p_dynamic"], float)
    unknown = state == 2
    reason = np.zeros(state.shape, np.uint8)
    def tag(code, predicate):
        selected = unknown & (reason == 0) & predicate
        reason[selected] = code

    tag(1, ~flow_valid)
    tag(2, evidence < cfg.min_evidence)
    tag(3, (qf < cfg.min_flow_reliability) & (qm < .5))
    tag(4, (np.abs(motion - map_score) > cfg.map_disagreement_threshold)
        & (qf > .5) & (qm > .5))
    tag(5, (p > cfg.static_threshold) & (p < cfg.dynamic_threshold))
    if np.any(unknown & (reason == 0)):
        raise AssertionError("Unexplained Unknown: V4.1a rule drift or NaN")
    if np.any(~unknown & (reason != 0)):
        raise AssertionError("Reason assigned to decided pixel")
    cues = {
        "map_uncovered": ~np.asarray(raw["pre_map_covered"], bool),
        "depth_invalid": ~np.asarray(raw["pre_depth_valid"], bool),
        "boundary": np.asarray(raw["boundary_near"], bool),
        "possible_occlusion_proxy": np.asarray(raw["possible_occlusion"], bool),
    }
    return reason, cues


def summarize_reasons(reason, cues):
    unknown = reason != 0
    out = {"unknown_fraction": float(unknown.mean()),
           "unknown_n": int(unknown.sum())}
    for k, name in REASONS.items():
        if k == 0:
            continue
        n = int(np.count_nonzero(reason == k))
        out["reason_" + name.lower() + "_n"] = n
        out["reason_" + name.lower() + "_fraction_image"] = n / reason.size
        out["reason_" + name.lower() + "_fraction_unknown"] = (
            n / int(unknown.sum()) if np.any(unknown) else float("nan"))
    for key, mask in cues.items():
        out["unknown_and_" + key + "_n"] = int((unknown & mask).sum())
        out["unknown_and_" + key + "_fraction_unknown"] = (
            float((unknown & mask).sum() / unknown.sum())
            if unknown.any() else float("nan"))
    if sum(out["reason_" + REASONS[i].lower() + "_n"]
           for i in range(1, 6)) != out["unknown_n"]:
        raise AssertionError("Reason categories must partition Unknown")
    return out


def validate_labels(gt, shape):
    """Require grayscale labels 0=static,1=dynamic,255=ignore."""
    arr = np.asarray(gt)
    if arr.ndim != 2 or arr.shape != shape:
        raise ValueError("GT must be [H,W] at original pixel resolution")
    values = np.unique(arr)
    if not np.isin(values, [0, 1, 255]).all():
        raise ValueError("GT must use EXACTLY 0,1,255; got %s" % values)
    if not np.isin(arr, [0, 1]).any():
        raise ValueError("No labeled pixels; all pixels are Ignore")
    return np.asarray(arr, np.uint8)


def predictions_from_raw(raw, shadow):
    """Baseline is original final-static inverse, always binary."""
    return {
        "legacy_binary": (~np.asarray(raw["final_static"], bool)).astype(np.uint8),
        **{key: np.asarray(shadow[key], np.uint8) for key in STATE_KEYS
           if key != "legacy_binary"},
    }


def confusion(gt, predicted, extra_accept=None):
    """Selective metrics only on labeled GT and accepted predictions.

    Effective dynamic recall uses ALL labeled dynamic pixels in denominator;
    abstentions contribute to missed dynamic detections. A penalized risk
    explicitly scores abstention as an error of cost 1. We report coverage.
    """
    gt = validate_labels(gt, np.asarray(predicted).shape)
    predicted = np.asarray(predicted)
    if not np.isin(np.unique(predicted), [0, 1, 2]).all():
        raise ValueError("Prediction must be 0/1/2")
    labeled = gt != 255
    accept = labeled & (predicted != 2)
    if extra_accept is not None:
        mask = np.asarray(extra_accept, bool)
        if mask.shape != gt.shape:
            raise ValueError("Extra acceptance mask shape mismatch")
        accept &= mask
    n_labeled = int(labeled.sum())
    n_accept = int(accept.sum())
    dynamic_total = int(((gt == 1) & labeled).sum())
    true_static_total = int(((gt == 0) & labeled).sum())
    tp = int((accept & (predicted == 1) & (gt == 1)).sum())
    fp = int((accept & (predicted == 1) & (gt == 0)).sum())
    fn = int((accept & (predicted == 0) & (gt == 1)).sum())
    tn = int((accept & (predicted == 0) & (gt == 0)).sum())
    return dict(n_gt=n_labeled, n_accepted=n_accept, n_abstained=n_labeled-n_accept,
                n_gt_dynamic=dynamic_total, n_gt_static=true_static_total,
                tp=tp, fp=fp, fn=fn, tn=tn)


def _ratio(a, b):
    return float(a / b) if b else float("nan")


def metrics_from_counts(c):
    tp, fp, fn, tn = (c[k] for k in ("tp", "fp", "fn", "tn"))
    n = c["n_gt"]
    n_dec = c["n_accepted"]
    coverage = _ratio(n_dec, n)
    precision = _ratio(tp, tp + fp)
    recall_decided = _ratio(tp, tp + fn)
    f1_decided = _ratio(2*tp, 2*tp + fp + fn)
    iou_decided = _ratio(tp, tp + fp + fn)
    gt_dyn = c["n_gt_dynamic"]
    return {
        **c,
        "coverage": coverage,
        "selective_risk": _ratio(fp+fn, n_dec),
        "selective_accuracy": _ratio(tp+tn, n_dec),
        "precision_decided": precision,
        "recall_decided": recall_decided,
        "f1_decided": f1_decided,
        "iou_decided": iou_decided,
        "recall_all_labeled_dynamic": _ratio(tp, gt_dyn),
        "abstain_on_gt_dynamic_fraction": _ratio(
            gt_dyn - tp - fn, gt_dyn),
        "abstain_on_gt_static_fraction": _ratio(
            c["n_gt_static"] - fp - tn, c["n_gt_static"]),
        "risk_abstention_cost_1": _ratio(
            fp+fn+c["n_abstained"], n),
    }


def eval_frame(raw, shadow, gt):
    pred = predictions_from_raw(raw, shadow)
    return {name: metrics_from_counts(confusion(gt, state))
            for name, state in pred.items()}


def combine_confusions(items):
    """Micro average across labeled pixels; frames are NOT independent pixels."""
    if not items:
        raise ValueError("No confusion sets to aggregate")
    keys=("n_gt","n_accepted","n_abstained","n_gt_dynamic",
          "n_gt_static","tp","fp","fn","tn")
    return metrics_from_counts({key: sum(x[key] for x in items) for key in keys})


def confidence_map(key, shadow):
    """Filtering confidence is predetermined; NO GT-dependent threshold fit."""
    if key == "legacy_binary":
        return None
    if key == "state_flow_only":
        score = np.asarray(shadow["motion_score"], float)
    elif key == "state_naive_rgbd":
        score = np.asarray(shadow["naive_score"], float)
    else:
        score = np.asarray(shadow["p_dynamic"], float)
    return np.clip(2*np.abs(score-.5), 0, 1)


def selective_curve(frames, thresholds=None):
    """Per-method coverage/risk sweep, preserving existing Unknown masks.

    frames: iterable (raw, shadow, independent_gt). Legacy baseline displayed
    as fixed point at coverage=1. Confidence is a score margin, NOT calibrated.
    """
    if thresholds is None:
        thresholds = np.linspace(0, .95, 20)
    out=[]
    for key in STATE_KEYS:
        for t in ([0.] if key == "legacy_binary" else thresholds):
            sets=[]
            for raw,shadow,gt in frames:
                pred=predictions_from_raw(raw,shadow)[key]
                conf=confidence_map(key,shadow)
                accept=None if conf is None else np.isfinite(conf) & (conf >= t)
                sets.append(confusion(gt,pred,accept))
            row=combine_confusions(sets)
            row.update(method=key,confidence_cutoff=float(t))
            out.append(row)
    return out
