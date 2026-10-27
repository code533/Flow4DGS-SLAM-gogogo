"""V4.1a offline Shadow model: motion, flow correspondence, map support.

Input is the EXISTING V4.0b .npz. This module never touches Tracking,
Mapping, optimizer or Gaussian state.

IMPORTANT: 'p_dynamic' and Q_F/Q_M are interpretable heuristic SCORES, not
calibrated posterior probabilities or validated uncertainty estimates.
No semantic mask or legacy dynamic mask participates in prediction.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class ShadowConfig:
    motion_floor_px: float = 0.5
    motion_scale_px: float = 2.5
    fb_scale_px: float = 1.0
    rgb_floor: float = 0.04
    rgb_scale: float = 0.15
    depth_floor_m: float = 0.05
    depth_scale_m: float = 0.35
    map_opacity_low: float = 0.40
    map_opacity_high: float = 0.95
    boundary_flow_factor: float = 0.65
    occlusion_flow_factor: float = 0.15
    unsupported_flow_factor: float = 0.35
    boundary_map_factor: float = 0.65
    flow_weight: float = 0.65
    map_weight: float = 0.35
    min_evidence: float = 0.15
    dynamic_threshold: float = 0.65
    static_threshold: float = 0.35
    min_flow_reliability: float = 0.25
    map_disagreement_threshold: float = 0.75

    def __post_init__(self):
        positive = ("motion_scale_px", "fb_scale_px", "rgb_scale",
                    "depth_scale_m")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError("%s must be positive" % name)
        if not (0 <= self.map_opacity_low < self.map_opacity_high <= 1):
            raise ValueError("Require 0 <= opacity_low < opacity_high <= 1")
        if not (0 <= self.static_threshold < self.dynamic_threshold <= 1):
            raise ValueError("Invalid state thresholds")
        for name in ("boundary_flow_factor", "occlusion_flow_factor",
                     "unsupported_flow_factor", "boundary_map_factor",
                     "flow_weight", "map_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError("%s must be in [0,1]" % name)
        if self.flow_weight + self.map_weight <= 0:
            raise ValueError("Nonzero evidence weights required")


REQUIRED_KEYS = (
    "flow_full_residual_px", "flow_full_valid", "fb_error_px", "fb_valid",
    "pre_opacity", "pre_map_covered", "pre_depth_valid",
    "pre_render_depth_valid", "pre_rgb_residual", "pre_depth_raw",
    "boundary_near", "possible_occlusion", "temporal_depth_support",
    "prior_static", "flow_dynamic", "final_static",
)


def _load_maps(m):
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        raise ValueError("V4.0b NPZ missing keys: %s" % ", ".join(missing))
    shape = np.asarray(m["flow_full_residual_px"]).shape
    if len(shape) != 2:
        raise ValueError("flow residual must have shape [H,W]")
    for key in REQUIRED_KEYS:
        if np.asarray(m[key]).shape != shape:
            raise ValueError("%s has incorrect shape" % key)
    return shape


def _nonnegative_score(residual, valid, floor, scale):
    """Map positive residual to [0,1]; invalid values receive zero EVIDENCE."""
    r = np.asarray(residual, dtype=np.float32)
    selected = np.asarray(valid, bool) & np.isfinite(r)
    score = np.zeros(r.shape, dtype=np.float32)
    score[selected] = -np.expm1(-np.maximum(r[selected] - floor, 0) / scale)
    return np.clip(score, 0, 1)


def _fraction(mask, denominator=None):
    mask = np.asarray(mask, bool)
    if denominator is None:
        return float(mask.mean())
    d = np.asarray(denominator, bool)
    return float((mask & d).sum() / d.sum()) if d.any() else float("nan")


def _avg(values, selected):
    x = np.asarray(values, np.float32)
    selection = np.asarray(selected, bool) & np.isfinite(x)
    return float(np.mean(x[selection])) if selection.any() else float("nan")


def infer_shadow(m, cfg=None):
    """Return scores and masks for four ablations using EXACTLY one NPZ.

    States: 0=Static, 1=Dynamic, 2=Unknown.
    Prior masks are passed through for *agreement diagnostics only*.
    Scores are uncalibrated; no labels or model fitting occurs here.
    """
    if cfg is None:
        cfg = ShadowConfig()
    shape = _load_maps(m)

    def arr(key, dtype=np.float32):
        return np.asarray(m[key], dtype=dtype)
    flow = arr("flow_full_residual_px")
    flow_valid = arr("flow_full_valid", bool) & np.isfinite(flow)
    fb = arr("fb_error_px")
    fb_valid = arr("fb_valid", bool) & np.isfinite(fb)
    boundary = arr("boundary_near", bool)
    possible_occ = arr("possible_occlusion", bool)
    support = arr("temporal_depth_support", bool)
    depth_valid = arr("pre_depth_valid", bool)
    render_valid = arr("pre_render_depth_valid", bool)
    opacity = np.clip(np.nan_to_num(arr("pre_opacity"), nan=0), 0, 1)
    rgb = arr("pre_rgb_residual")
    depth = arr("pre_depth_raw")
    rgb_valid = np.isfinite(rgb)
    depth_resid_valid = depth_valid & render_valid & np.isfinite(depth)

    # Motion state evidence: disagreement with static-camera motion.
    motion_score = _nonnegative_score(
        flow, flow_valid, cfg.motion_floor_px, cfg.motion_scale_px)

    # Correspondence reliability: FB agreement, boundary, occlusion-proxy.
    q_flow = np.zeros(shape, dtype=np.float32)
    q_flow[fb_valid] = np.exp(-np.maximum(fb[fb_valid], 0) /
                               cfg.fb_scale_px)
    q_flow *= np.where(boundary, cfg.boundary_flow_factor, 1.0)
    q_flow *= np.where(possible_occ, cfg.occlusion_flow_factor, 1.0)
    q_flow *= np.where(support, 1.0, cfg.unsupported_flow_factor)
    q_flow *= flow_valid.astype(np.float32)
    q_flow = np.clip(q_flow, 0, 1)

    # Map support is NOT the same thing as map residual / motion status.
    q_map = np.clip((opacity - cfg.map_opacity_low) /
                    (cfg.map_opacity_high - cfg.map_opacity_low), 0, 1)
    q_map *= (depth_valid & render_valid).astype(np.float32)
    q_map *= np.where(boundary, cfg.boundary_map_factor, 1.0)
    q_map = q_map.astype(np.float32)

    # Multi-modal discrepancy is only evidence where map rendering exists.
    rgb_score = _nonnegative_score(rgb, rgb_valid,
                                   cfg.rgb_floor, cfg.rgb_scale)
    depth_score = _nonnegative_score(depth, depth_resid_valid,
                                     cfg.depth_floor_m, cfg.depth_scale_m)
    rgb_weight = (rgb_valid & depth_valid & render_valid).astype(np.float32)
    depth_weight = depth_resid_valid.astype(np.float32)
    map_denom = 0.4 * rgb_weight + 0.6 * depth_weight
    map_score = np.divide(
        0.4 * rgb_weight * rgb_score + 0.6 * depth_weight * depth_score,
        map_denom, out=np.zeros(shape, np.float32), where=map_denom > 0)
    q_map *= (map_denom > 0).astype(np.float32)

    # A0 flow-only score; A1 naive fusion; A2 reliability-weighted binary;
    # A3 reliability-weighted 3-state (proposed Shadow prototype).
    naive_weight = 0.6 * flow_valid.astype(np.float32) + 0.4 * (map_denom > 0)
    naive = np.divide(
        0.6 * flow_valid * motion_score + 0.4 * (map_denom > 0) * map_score,
        naive_weight, out=np.zeros(shape, np.float32), where=naive_weight > 0)

    flow_evidence = cfg.flow_weight * q_flow
    map_evidence = cfg.map_weight * q_map
    evidence = flow_evidence + map_evidence
    fused = np.divide(flow_evidence * motion_score + map_evidence * map_score,
                      evidence, out=np.zeros(shape, np.float32),
                      where=evidence > 0)
    fused = np.clip(fused, 0, 1)

    # If observation evidence is absent, do NOT quietly decide "Static".
    # High model discrepancy with low support is explicitly Unknown.
    disagreement = (np.abs(motion_score - map_score) >
                    cfg.map_disagreement_threshold) & (
                    q_flow > 0.5) & (q_map > 0.5)
    undecidable = (
        ~flow_valid |
        (evidence < cfg.min_evidence) |
        ((q_flow < cfg.min_flow_reliability) & (q_map < 0.5)) |
        disagreement)
    three_state = np.full(shape, 2, dtype=np.uint8)
    three_state[(~undecidable) & (fused <= cfg.static_threshold)] = 0
    three_state[(~undecidable) & (fused >= cfg.dynamic_threshold)] = 1

    flow_state = np.full(shape, 2, dtype=np.uint8)
    flow_state[flow_valid] = (motion_score[flow_valid] >=
                              cfg.dynamic_threshold).astype(np.uint8)
    naive_state = np.full(shape, 2, dtype=np.uint8)
    naive_state[naive_weight > 0] = (
        naive[naive_weight > 0] >= cfg.dynamic_threshold).astype(np.uint8)
    binary_state = np.full(shape, 2, dtype=np.uint8)
    binary_state[evidence > 0] = (
        fused[evidence > 0] >= cfg.dynamic_threshold).astype(np.uint8)

    return {
        "p_dynamic": fused.astype(np.float32),
        "q_flow": q_flow,
        "q_map": q_map,
        "motion_score": motion_score,
        "map_score": map_score.astype(np.float32),
        "naive_score": naive.astype(np.float32),
        "evidence": evidence.astype(np.float32),
        "state_flow_only": flow_state,
        "state_naive_rgbd": naive_state,
        "state_reliability_binary": binary_state,
        "state_decoupled_3state": three_state,
        "unknown_from_missing_evidence": undecidable,
        "flow_valid": flow_valid,
        "map_covered": arr("pre_map_covered", bool),
    }


def summarize_shadow(m, o):
    """Agreement with ORIGINAL MASK is descriptive, NEVER detection accuracy."""
    prior = np.asarray(m["prior_static"], bool)
    baseline_dyn = ~np.asarray(m["final_static"], bool)
    valid = np.asarray(o["flow_valid"], bool)
    state = o["state_decoupled_3state"]
    decided = state != 2
    row = {
        "frame_id": int(np.asarray(m["frame_id"]).item()),
        "n_pixels": int(state.size),
        "flow_valid_fraction": _fraction(valid),
        "map_covered_fraction": _fraction(o["map_covered"]),
        "three_state_static_fraction": _fraction(state == 0),
        "three_state_dynamic_fraction": _fraction(state == 1),
        "three_state_unknown_fraction": _fraction(state == 2),
        "unknown_in_map_uncovered": _fraction(
            state == 2, ~np.asarray(o["map_covered"], bool)),
        "unknown_in_map_covered": _fraction(
            state == 2, np.asarray(o["map_covered"], bool)),
        "mean_q_flow_prior_dynamic": _avg(o["q_flow"], ~prior & valid),
        "mean_q_flow_prior_static": _avg(o["q_flow"], prior & valid),
        "mean_q_map_covered": _avg(o["q_map"], o["map_covered"]),
        "mean_evidence": _avg(o["evidence"], valid),
        "legacy_dynamic_fraction": _fraction(baseline_dyn),
        "agreement_with_legacy_decided_NOT_accuracy": _fraction(
            (state == 1) == baseline_dyn, decided),
    }
    for key in ("state_flow_only", "state_naive_rgbd",
                "state_reliability_binary", "state_decoupled_3state"):
        st = o[key]
        decided_variant = st != 2
        row[key + "_coverage_fraction"] = _fraction(decided_variant)
        row[key + "_dynamic_fraction"] = _fraction(st == 1)
        row[key + "_agreement_with_legacy_NOT_accuracy"] = _fraction(
            (st == 1) == baseline_dyn, decided_variant)
    return row


def config_as_dict(cfg):
    return asdict(cfg)
