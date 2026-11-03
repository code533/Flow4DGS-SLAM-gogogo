# Final controlled verification: stop/go gate for V4.3a.4 candidate scoring

**Goal:** End the G0/Support/G4 candidate-score development line using
**one prespecified cross-sequence test**, not another round of hyperparameter
tuning. This branch is based on `feature/v43a4-simple-config`.

## Primary study

- **Development sequence:** `rgbd_bonn_balloon2`, V4.3a.4 results already
  available. Not used to tune on the validation sequence.
- **Independent validation sequence:** `rgbd_bonn_balloon` (a different
  sequence in the same Bonn dataset, not a completely independent dataset).
- Fixed sampling: first eligible source propagation event and every fourth
  after; maximum 25 captured sources.
- Next arriving keyframe: at most 12 frames later. Same pre-source-Mapping
  Gaussian map is frozen for all candidates; static pose and observed
  future RGB-D are matched. Source displacements use zero-order hold;
  this is **NOT** full future motion prediction.
- **Locked PRIMARY** switch margin = **0**, comparing ordinary dynamic
  depth MAE, 50mm capped residual, Huber20 and 5px-eroded interior capped
  residual. The original 0.2mm threshold is an explicitly SECONDARY
  sensitivity evaluation on the same candidate scores.
- Candidate choices computed from SOURCE scores only. All choices are
  evaluated on the **same FUTURE ordinary dynamic depth MAE**. Always-G0
  is the negative control.
- Capping is `mean(min(error_m,0.05m))` per candidate: NOT capping paired
  differences. Huber has a linear, unbounded tail.
- Source selection coverage tolerance 0.5 percentage points; at least
  95% source AND future depth coverage and 256 observed dynamic depth
  pixels. Prefer G0 unless the candidate has a strictly lower score.
- No additional search or tuning on this heldout scene. Record missing
  or failed cases, do not change scene or rules in response.

The prior balloon2 0mm-margin result was an *exploratory* observation;
it cannot count as independent confirmation. We lock it now BEFORE
obtaining the new balloon-sequence results.

## Run commands (from repo root)

```bash
git fetch origin
git switch feature/v43a4-final-heldout-control
git pull --ff-only

# Checks do not require a GPU.
python scripts/check_v43a4_final_holdout.py --check-dataset-path
python -m unittest discover -s tests -p 'test_v43a4_final_controlled.py' -v
python -m unittest discover -s tests -p 'test_v43a4_predictive.py' -v

# First and ONLY planned independent-scene test:
python slam.py \
  --config configs/rgbd/bonn/v43a4_final_balloon_holdout.yaml \
  --eval --dynamic --exp_name=v43a4_final_balloon_holdout --save_results=0
```

DO NOT use `--inherit_from`; the YAML inherits the prebuilt clean
Bonn V4.3a.4 preset. Do not use `ballon2.yaml` or `balloon2` here.

The SLAM run stores its merged, actual config as:
`<run>/config.yml`. Keep that exact file in the result package.

### Locked analysis

```bash
python scripts/v43a4_final_controlled.py \
  --holdout-scores "<run>/v43a4_predictive/v43a4_candidate_scores.csv" \
  --holdout-config "<run>/config.yml" \
  --development-scores "/path/to/previous/v43a4_candidate_scores.csv" \
  --output "<run>/v43a4_final_decision"
```

The development scores are OPTIONAL CONTEXT ONLY and cannot affect
the heldout primary decision. If unavailable, omit the argument.

Output:
- `v43a4_final_policy_per_pair.csv`: policy choices on all pairs,
  including PRIMARY 0mm and secondary 0.2mm margin.
- `v43a4_final_policy_summary.csv`: comparable policy statistics.
- `v43a4_final_mae_vs_clipped50.csv`: direct paired difference for each
  source/future pair.
- `v43a4_final_protocol_and_decision.json`: explicit locked thresholds,
  SHA256 of observed inputs, and decision.
- `v43a4_final_decision.md`: one-page stop/go decision with rationale.

## Strict prespecified criteria

Main question: Is **Clipped50** systematically better than ordinary
MAE for predicting a future better motion hypothesis?

The locked screening gate requires:
- >=12 valid pairs, >=5 **different** MAE-versus-Clipped decisions.
- At least **+0.5mm** mean future gain for Clipped50 versus MAE,
  measured per time pair on future raw depth MAE.
- Median paired gain not worse than -0.1mm, and near-future-oracle
  frequency at least **+5 percentage points** versus MAE.
- No extra future RGB deterioration >0.0005 or candidate coverage
  deterioration >0.5 percentage points.
- Positive mean future gain for Clipped50 relative to always-G0.

Decision:
- `NO_GO_STOP_SCORE_SELECTION` if adequately powered descriptive
  screen fails one or more gates. Redirect to direct occlusion-aware
  Gaussian motion estimation; NO extra scoring hyperparameter tuning.
- `INSUFFICIENT_DATA_NO_GO` if fewer than 12 heldout pairs. Evidence
  insufficient; do not present a positive candidate-selection claim or
  retrospectively optimize source scores.
- `PASS_ONLY_TO_ONLINE_ABLATION` if it passes all gates.
  Still **not** a published improvement; repeat across independent
  sequences/runs and evaluate real online Mapping + metrics before any
  publication claim.

This is a decision-oriented screening rule, not a formal power analysis
or a statistically independent significance test. Different source
keyframes in a single Bonn sequence are correlated. Report per-pair
traces instead of binomial p-values. The frozen-state MAE is not ATE,
PSNR/LPIPS nor physical 3D flow ground truth.

## Important original implementation constraints

- A single snapshot of GaussianModel can occupy significant GPU RAM.
  The existing V4.3a.4 code retains at most one pending snapshot, but
  rendering may still affect scheduling and FPS. Same config and model
  across all policies; do not interpret the diagnostic run ATE as
  an online algorithm improvement.
- Keyframe selection may differ stochastically between runs.
- There is no need to rerun the entire V4.3a.1/V4.3a.2 diagnostic suite.
  The heldout experiment reuses the V4.3a.4 source/next-keyframe
  candidate-score generation and the *new* fixed analysis.
- If the balloon sequence is not present locally, the validation
  cannot be run. Do not silently replace it with balloon2; supply a
  different predeclared, supported dynamic sequence instead.
