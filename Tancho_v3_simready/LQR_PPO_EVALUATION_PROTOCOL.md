# Tancho V3 Common Balance Evaluation Protocol

This protocol is controller-independent. LQR and PPO must use the same frozen
physics asset, reset implementation, terrain, friction, time step, torque limit,
initial states, termination thresholds, logging schema, and metric script.

## Initial conditions

- Fixed-leg pose: thigh `-0.50 rad`, calf `+0.87 rad`
- Initial pitch: `-10, -7, -5, -3, -1, +1, +3, +5, +7, +10 deg`
- Initial root, joint, and wheel velocities: zero
- Wheel torque limit: `±0.45 Nm`
- Flat terrain; no pushes or observation noise
- At least 10 deterministic seeds per initial pitch
- Episode duration: 10 s unless failure occurs first
- Failure pitch: `|pitch| >= 15 deg`, non-wheel contact, or numerical failure

## Required CSV columns

```text
time_s,pitch_rad,pitch_rate_rad_s,base_x_m,
wheel_L_q_rad,wheel_R_q_rad,
wheel_L_qd_rad_s,wheel_R_qd_rad_s,
wheel_L_qdd_rad_s2,wheel_R_qdd_rad_s2,
wheel_L_tau_Nm,wheel_R_tau_Nm
```

Each CSV must also have a sidecar JSON containing controller name/version,
initial pitch, seed, URDF SHA-256, physics-baseline version, dt, friction, and
solver settings.

## Common metrics

Run `scripts/evaluation/balance_metrics.py` for every episode and aggregate by
initial pitch. Report success/failure, pitch RMS, maximum pitch, settling time,
base displacement, wheel torque RMS/peak, saturation percentage, and maximum
wheel speed. Report median plus 5th/95th percentiles; never compare only the
best seed.

No controller-specific reset, torque clipping, filtering, early termination,
or hidden warm-up step is allowed.
