# Tancho V3 Fixed Flat — Experimental-variable audit

Task: `TanchoV3-Fixed-Flat-v0`

This file separates API/plumbing choices from settings that must be disclosed
when results are reported in a paper.  The fixed task is not presented as a
drop-in controller change: it is a different, explicitly identified plant.

## Intended fixed-vs-full variables (must report)

- Asset: `Tancho_v3_fixed.urdf` instead of `Tancho_v3.urdf`.
- Morphology: thigh/calf bodies are rigidly merged at thigh `-0.50 rad`, calf
  `+0.87 rad`; only `joint_wheel_L/R` remain movable.
- Merged root mass/COM/inertia are those recorded in `Tancho_v3_fixed.json`.
- Action space changes from six terms to two wheel-effort terms.
- Leg implicit-PD actuators are absent; they are not simulated with large gains.
- Wheel limits remain `0.45 Nm` and `188 rad/s`, with zero stiffness/damping.
- Joint-state and previous-action observation dimensions change from six to two.
- Fixed-body leg collisions are merged into `base_link_root`; consequently the
  shared `base_contact` termination also detects collision of the rigid legs.

## Shared controlled settings (must report for reproducibility)

- Flat terrain; static/dynamic friction `0.8/0.8`, restitution `0.0`, multiply
  combine modes.
- Physics `dt=0.005 s`, decimation `2`, control period `0.010 s`, episode `20 s`.
- Reward terms and weights: alive `+1.0`, flat orientation `-5.0`, XY angular
  velocity `-0.5`, XY linear velocity `-0.2`, wheel torque `-1e-4`.
- Terminations: root-body contact above `10 N`, orientation limit `0.85 rad`,
  and time-out.
- Observation terms and scales, command distribution, PPO configuration, seed,
  Isaac Lab/PhysX version, solver settings, and source revision.
- Mass and friction randomization currently use zero-width ranges (disabled).
  Push velocity is initially zero.
- Velocity-command and push curricula are scheduled at 36,000 manager steps;
  this equals the configured 1,500-iteration endpoint and must not be described
  as an active training phase unless logs show that it actually executed.

## Custom/non-MDP code classification

- `reset_tancho_on_wheels`: custom engineering implementation, but its output
  is an experimental initial-condition variable.  Report wheel phase `0`, zero
  root/joint velocity, geometry-derived root height, flat terrain height `0`,
  canonical TPU support geometry, and `0.05 mm` contact preload.
- `cr.lin_vel_xy_l2`: API-compatibility implementation of the mathematical term
  `||v_base,xy||^2`.  It replaces an unavailable Isaac Lab symbol without
  changing the reward definition; report the formula, not as a separate ablation.
- `cr.curriculum_enable_velocity` and `cr.curriculum_enable_push`: custom
  callbacks that alter the training distribution and therefore are paper
  variables whenever their thresholds are reached.
- Geometry/STL parsing, task registration, log experiment names, and diagnostic
  scripts are plumbing and are not scientific variables by themselves.
- `prepare_tancho_measurement_start` is not a training event.  If used during
  evaluation, its unrecorded contact warm-up and formal-t0 procedure must be
  disclosed in the evaluation protocol.

## Comparison rule

Do not compare an old full-model checkpoint with this fixed task and attribute
the difference only to fixed legs unless both runs use the same reward,
termination, reset, contact material, timing, PPO configuration, and source
revision.  Otherwise those changes are confounds.
