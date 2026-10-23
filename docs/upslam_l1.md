# Layer 1 — static-render uncertainty mask (experimental)

This is a **training-free RGB-D residual approximation inspired by UP-SLAM**.
It is *not* the full UP-SLAM algorithm (no DINO features, feature Gaussians, or uncertainty MLP).
It replaces the **motion-mask refinement** supplied by optical-flow residuals
in the front-end; the original optical-flow calculations continue to serve
camera motion fitting and back-end dynamic Gaussian propagation/mapping.

## Run

Use `configs/rgbd/bonn/ballon2_upslam_l1.yaml` using:

```bash
python slam.py --config configs/rgbd/bonn/ballon2_upslam_l1.yaml --exp_name upslam_l1
```

For the baseline, switch `upslam_l1.enabled: false` and use a different
`--exp_name`.

For a controlled A/B comparison run the **same** config twice:
- Baseline: set `model_params.upslam_l1.enabled: false`.
- L1: set `model_params.upslam_l1.enabled: true`.

Retain `flow_mask: true` and `flow_cam: true` for the initial ablation.
These preserve the legacy flow-based camera fit; the dynamic mask no longer
uses the residual-flow segmentation decision when L1 is enabled.

Each frame prints `[UPSLAM-L1]` status, static-render coverage, and fraction
of *valid* pixels newly rejected as inconsistent.
If `save_diagnostics: true`, per-frame NPZ files are in the output
`tracking/upslam_l1_XXXXX.npz` directory:
`rgb_err`, `depth_err`, `valid`, `candidate`, `final_static`,
`prior_static`, `coverage`, `rejection_ratio`, `status`.

Conventions: `motion_mask=True` means static; `False` means dynamic.

## Safeguards and limitations

- Render uses only Gaussians flagged static `dygs == False`, at the
  *initial* camera pose copied from the previous frame.
- Low-opacity/invalid-depth/uncovered pixels are **not** labeled dynamic;
  prior mask is retained.
- The default uses RGB **AND** depth residual thresholds to reduce false positives.
- Low coverage and excessive rejections trigger fallback to the input mask.
- The initial mask may contain upstream YOLO-derived priors; disabling the
  residual-flow refinement **does not** remove that segmentation source.
- Camera-motion-induced residuals can still cause false-positive dynamic
  pixels. This is a first-level hypothesis test, not pose-invariant motion
  detection. Increase thresholds or improve pose initialization if needed.
- It is normal for ATE to fluctuate. Compare multiple seeds/runs and inspect
  per-frame coverage, dynamic fraction and visualization before claiming gains.
- Do not use `flow_mask: false` as the L1-only A/B baseline, because in the
  original code that also bypasses flow-camera initialization.
- The first frame remains controlled by the original initialization and mask.
- The supplied config inherits the tracked Bonn `config.yaml`, not the
  scene-specific `ballon2.yaml` which references `base_config.yaml`
  that is missing in this snapshot of the main tree. If your local
  baseline uses a different config, apply only the `upslam_l1` block
  to that baseline and use it for both runs.

## Success criteria

Check for nontrivial accepted masks (not only fallbacks), plausible spatial
localization of rejected regions, stable tracking, and ATE relative to a
matching baseline. This layer does **not** claim elimination of RAFT.
