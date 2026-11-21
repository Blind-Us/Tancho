#!/usr/bin/env python3
"""Zero-command stand / push check of an exported 6-DOF policy in MuJoCo (no Isaac).

Builds the robot from the URDF (fixed links fused, free root), a flat floor with the
Isaac effective tire friction 0.64, and the same motor laws as Isaac's implicit
actuators (MuJoCo servos under implicitfast, 1 ms; ``--motors explicit`` computes tau
every 1 ms instead, which chatters on the light wheel: dt*Kd/I ~ 200):
  legs   tau = 20 (q* - q) - 0.2 qdot,   |tau| <= 12.5 N m
  wheels tau = 4 (qdot* - qdot),         |tau| <= 0.45 N m
Policy at 50 Hz.  Observation layout from ``config.json`` of the hop r3 MuJoCo bundle;
29 inputs (HopPush / Stability) or 31 (+ wheel odometry r*q clipped +/-0.5 m, x2).
Not modelled: PhysX's 12.5 rad/s leg joint speed cap.

    python scripts/wheel_only/evaluate_stand_mujoco.py --checkpoint <run>/exported/policy.pt \
        --duration 60 --push-vel 0 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / "releases/tancho_v3_hop_r3_mujoco/robot/Tancho_v3.urdf"
ACT_JOINTS = ["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R", "joint_wheel_L", "joint_wheel_R"]
LEGS, WHEELS = ACT_JOINTS[:4], ACT_JOINTS[4:]
DEFAULT = {"joint_thigh_L": -0.5, "joint_calf_L": 0.87, "joint_thigh_R": -0.5, "joint_calf_R": 0.87}
LEG_SCALE, WHEEL_SCALE = 0.6, 52.3598776
KP, KD_LEG, LEG_TAU = 20.0, 0.2, 12.5
KD_WHEEL, WHEEL_TAU = 4.0, 0.45
LEG_VEL_LIMIT = 12.5
WHEEL_R = 0.03614
ROOT_Z = 0.2614 + 0.001
FRICTION = 0.64

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--duration", type=float, default=60.0)
parser.add_argument("--push-time", type=float, default=10.0)
parser.add_argument("--push-vel", type=float, default=0.0)
parser.add_argument("--presses", default="", help="e.g. 'LT:3,RT:6,LT+RT:9,CHG:12' (button:start s); LT/RT held --press-s, CHG held --charge-s")
parser.add_argument("--press-s", type=float, default=0.4)
parser.add_argument("--charge-s", type=float, default=0.6)
parser.add_argument("--charge-jump", type=int, default=1, help="0: released crouch stands back up (StationSquat), 1: charged hop (HopCharge)")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--motors", choices=["implicit", "explicit"], default="implicit",
                    help="implicit: MuJoCo servos (Kd integrated implicitly, like Isaac); explicit: tau computed every 1 ms")
parser.add_argument("--friction", type=float, default=FRICTION, help="tire / floor friction (Isaac nominal 0.8 x 0.8)")
parser.add_argument("--init-pitch-deg", type=float, default=0.0, help="initial body pitch offset (deg)")
parser.add_argument("--video", action="store_true", help="write <output-dir>/mujoco.mp4 (run with MUJOCO_GL=egl)")
parser.add_argument("--latency-ms", type=int, default=0, help="motor targets take effect this many ms after each policy step")
parser.add_argument("--obs-noise", action="store_true", help="add the training sensor noise (uniform) to the observation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--slope-deg", type=float, default=0.0, help="emulate an incline by tilting gravity about y (+x downhill)")
parser.add_argument("--dt", type=float, default=0.001, help="MuJoCo physics step (Isaac: 0.005)")
parser.add_argument("--wheel-kd", type=float, default=4.0, help="wheel velocity-loop Kd (StationV4: 1.0)")
parser.add_argument("--noslip", type=int, default=10, help="MuJoCo noslip_iterations (0 = MuJoCo default)")
parser.add_argument("--tire", choices=["cylinder", "sphere"], default="cylinder", help="tire collision shape (sphere: same radius)")
parser.add_argument("--solref", type=float, default=0.005, help="MuJoCo contact time constant (s)")
args = parser.parse_args()
DT = args.dt
KD_WHEEL = args.wheel_kd
SUB = round(0.02 / DT)  # 50 Hz policy


def build_model() -> mujoco.MjModel:
    text = URDF.read_text()
    meshdir = (URDF.parent / "meshes").as_posix()
    text = re.sub(r"(<robot[^>]*>)", rf'\1<mujoco><compiler meshdir="{meshdir}" discardvisual="true" fusestatic="true" balanceinertia="true"/></mujoco>', text, count=1)
    text = text.replace('filename="meshes/', 'filename="')
    spec = mujoco.MjSpec.from_string(text)
    spec.option.timestep = DT
    a = math.radians(args.slope_deg)
    spec.option.gravity = [9.81 * math.sin(a), 0.0, -9.81 * math.cos(a)]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Soft friction constraints let a loaded tire creep (2-3 cm/s here with the wheel
    # almost still); the no-slip solver pass removes it, as PhysX's rigid friction.
    spec.option.noslip_iterations = args.noslip
    root = spec.worldbody.first_body()
    root.add_freejoint()
    root.pos = [0, 0, ROOT_Z]
    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20, 20, 0.1]
    # Robot geoms touch only the floor (Isaac: no self-collision).
    for g in spec.geoms:
        g.friction = [args.friction, 0.005, 0.0001]
        g.contype, g.conaffinity = 2, 1
        g.solref = [args.solref, 1.0]
        if args.tire == "sphere" and g.name.endswith("tire_collision"):
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [WHEEL_R, 0, 0]
    floor.contype, floor.conaffinity = 1, 2
    # Video only: brighter scene (the URDF's visual meshes are discarded, collision shapes are drawn).
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    floor.rgba = [0.75, 0.78, 0.8, 1.0]
    for g in spec.geoms:
        if g is not floor:
            g.rgba = [0.15, 0.15, 0.15, 1.0] if g.name.endswith("tire_collision") else [0.35, 0.55, 0.85, 1.0]
    for j in ACT_JOINTS:
        a = spec.add_actuator()
        a.target = j
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        if args.motors == "explicit":
            a.gainprm[0] = 1.0
            a.ctrllimited = False
            continue
        # MuJoCo position / velocity servos: implicitfast integrates the Kd term
        # implicitly, like Isaac's implicit actuators.
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.ctrllimited = False
        a.forcelimited = True
        if j in LEGS:
            # Legs: torque computed below with PhysX's 12.5 rad/s joint speed cap
            # (light enough for an explicit 1 ms PD: dt*Kd/I < 1).
            a.biastype = mujoco.mjtBias.mjBIAS_NONE
            a.gainprm[0] = 1.0
            a.forcerange = [-LEG_TAU, LEG_TAU]
        else:
            a.gainprm[0] = KD_WHEEL
            a.biasprm[0:3] = [0.0, 0.0, -KD_WHEEL]
            a.forcerange = [-WHEEL_TAU, WHEEL_TAU]
    return spec.compile()


model = build_model()
data = mujoco.MjData(model)
jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ACT_JOINTS}
qadr = {n: model.jnt_qposadr[i] for n, i in jid.items()}
vadr = {n: model.jnt_dofadr[i] for n, i in jid.items()}
for n, q in DEFAULT.items():
    data.qpos[qadr[n]] = q
if args.init_pitch_deg:
    h = math.radians(args.init_pitch_deg) / 2  # about world y (+ nose down)
    data.qpos[3:7] = [math.cos(h), 0.0, math.sin(h), 0.0]
mujoco.mj_forward(model, data)
# Put the tires on the floor.
wheel_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ("wheel_L", "wheel_R")]
data.qpos[2] += WHEEL_R - min(data.xpos[b][2] for b in wheel_bodies) + 0.001
mujoco.mj_forward(model, data)

policy = torch.jit.load(str(args.checkpoint), map_location="cpu").eval()
obs_dim = None
for n_in in (34, 32, 31, 29):
    try:
        with torch.no_grad():
            policy(torch.zeros(1, n_in))
        obs_dim = n_in
        break
    except Exception:
        continue
assert obs_dim in (29, 31, 32, 34), f"unsupported policy input size {obs_dim}"

RX_T = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # body -> IMU (+90 deg about x)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def pitch_deg(q):
    w, x, y, z = q
    return math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def build_obs(last_action):
    R = quat_to_mat(data.qpos[3:7])
    g_b = R.T @ (model.opt.gravity / np.linalg.norm(model.opt.gravity))
    w_b = data.qvel[3:6].copy()  # free joint angular velocity is in the body frame
    o = [RX_T @ g_b, 0.25 * (RX_T @ w_b), np.zeros(3)]
    o.append(0.05 * np.array([data.qvel[vadr[j]] for j in WHEELS]))
    o.append(last_action)
    o.append(np.array([data.qpos[qadr[j]] - DEFAULT[j] for j in LEGS]))
    o.append(0.05 * np.array([data.qvel[vadr[j]] for j in LEGS]))
    o.append(trig.copy())  # LT / RT held
    o.append(np.minimum(phase, 0.6) / 0.6)
    if obs_dim in (31, 34):
        o.append(2.0 * np.clip(WHEEL_R * np.array([data.qpos[qadr[j]] for j in WHEELS]), -0.5, 0.5))
    if obs_dim in (32, 34):
        o.append(np.array([chg_held, chg_level, min(chg_phase, 0.6) / 0.6]))
    return np.concatenate(o).astype(np.float32)


# -- operator presses: LT / RT lift reference (Pi table) and the HopCharge crouch -------
LIFT = np.loadtxt(ROOT / "releases/tancho_v3_hop_r3_mujoco/reference_lift_table.csv", delimiter=",", skiprows=1)[:, 2:4]
CROUCH, CROUCH_S, EXTEND, CHARGE_STEP_S = np.array([-0.305, 0.75]), 0.35, np.array([0.245, -0.60]), 0.01
CHARGE_KEYS = ((CHARGE_STEP_S, 0.245, -0.60), (0.07, 0.245, -0.60), (0.08, -0.305, 0.75), (0.16, -0.305, 0.75), (0.28, 0.0, 0.0))
trig, phase = np.zeros(2), np.full(2, 10.0)
chg_held, chg_level, chg_phase, chg_release_level = 0.0, 0.0, 10.0, 0.0
PRESSES = [(b, float(t)) for b, t in (x.split(":") for x in args.presses.split(",") if x)]


def lift_offset(p):
    i = int(round(p / 0.02))
    return LIFT[i] if 0 <= i < len(LIFT) else np.zeros(2)


def keyframes(p, keys):
    for (t0, a0, b0), (t1, a1, b1) in zip(keys[:-1], keys[1:]):
        if t0 <= p < t1:
            w = (p - t0) / (t1 - t0)
            return np.array([a0, b0]) * (1 - w) + np.array([a1, b1]) * w
    return np.zeros(2)


def charge_offset():
    if chg_held > 0:
        return chg_level * CROUCH
    if not args.charge_jump:
        return chg_release_level * CROUCH * max(0.0, 1.0 - chg_phase / CHARGE_KEYS[-1][0])
    if chg_phase < CHARGE_STEP_S:
        w = chg_phase / CHARGE_STEP_S
        return chg_release_level * CROUCH * (1 - w) + EXTEND * w
    return keyframes(chg_phase, CHARGE_KEYS)


def update_buttons(t):
    """Sides held at time t; rising edges start that side's 0.4 s lift; CHG as ChargeCommand."""
    global trig, phase, chg_held, chg_level, chg_phase, chg_release_level
    new = np.zeros(2)
    new_chg = 0.0
    for b, t0 in PRESSES:
        if b == "CHG":
            new_chg = max(new_chg, float(t0 <= t < t0 + args.charge_s))
        elif t0 <= t < t0 + args.press_s:
            if "LT" in b:
                new[0] = 1.0
            if "RT" in b:
                new[1] = 1.0
    rising = (new > 0) & (trig <= 0) & (phase >= 0.399)
    phase = np.where(rising, 0.0, phase + 0.02)
    trig = new
    released = new_chg <= 0 and chg_held > 0
    if released:
        chg_release_level = chg_level
    chg_phase = 0.0 if released else chg_phase + 0.02
    chg_level = min(chg_level + 0.02 / CROUCH_S, 1.0) if new_chg > 0 else 0.0
    chg_held = new_chg


last = np.zeros(6, dtype=np.float32)
rng = np.random.default_rng(args.seed)
# Training sensor noise (Isaac adds it before the scale): gravity, gyro, wheel speed, leg pos, leg speed.
NOISE = np.zeros(34 if obs_dim >= 32 else obs_dim, np.float32)
NOISE[0:3], NOISE[3:6], NOISE[9:11], NOISE[17:21], NOISE[21:25] = 0.02, 0.05 * 0.25, 0.2 * 0.05, 0.01, 0.2 * 0.05
NOISE = NOISE[:obs_dim]
LAT_SUB = int(round(args.latency_ms / (DT * 1000)))
q_prev = v_prev = None
xy0 = data.qpos[0:2].copy()
p0 = pitch_deg(data.qpos[3:7])
steps = math.ceil(args.duration / (DT * SUB))
push_step = round(args.push_time / (DT * SUB)) if args.push_vel > 0 else None
rows, fell_at, sat = [], None, []
frames = []
if args.video:
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance, cam.azimuth, cam.elevation = 1.1, 135.0, -12.0
for k in range(steps):
    if push_step is not None and k == push_step:
        R = quat_to_mat(data.qpos[3:7])
        fwd = R[:, 0].copy(); fwd[2] = 0; fwd /= np.linalg.norm(fwd)
        data.qvel[0:2] += args.push_vel * fwd[:2]
    update_buttons(k * DT * SUB)
    with torch.no_grad():
        o = build_obs(last)
        if args.obs_noise:
            o = o + rng.uniform(-NOISE, NOISE).astype(np.float32)
        a = policy(torch.from_numpy(o).unsqueeze(0))[0].numpy()
    a = np.clip(a, -1.0, 1.0)
    last = a.astype(np.float32)
    off = np.concatenate([lift_offset(phase[0]), lift_offset(phase[1])])
    if obs_dim in (32, 34):
        off = off + np.tile(charge_offset(), 2)
    q_t = {j: DEFAULT[j] + LEG_SCALE * a[i] + off[i] for i, j in enumerate(LEGS)}
    v_t = {j: WHEEL_SCALE * a[4 + i] for i, j in enumerate(WHEELS)}
    nsat = 0
    q_new, v_new = q_t, v_t
    for sub in range(SUB):
        if q_prev is not None and sub < LAT_SUB:
            q_t, v_t = q_prev, v_prev
        else:
            q_t, v_t = q_new, v_new
        for i, j in enumerate(ACT_JOINTS):
            q, qd = data.qpos[qadr[j]], data.qvel[vadr[j]]
            if j in q_t:
                tau = np.clip(KP * (q_t[j] - q) - KD_LEG * qd, -LEG_TAU, LEG_TAU)
                if abs(qd) >= LEG_VEL_LIMIT and tau * qd > 0:
                    tau = 0.0
            else:
                tau = np.clip(KD_WHEEL * (v_t[j] - qd), -WHEEL_TAU, WHEEL_TAU)
                if args.motors == "explicit":
                    nsat += abs(tau) >= 0.95 * WHEEL_TAU
            data.ctrl[i] = tau if (args.motors == "explicit" or j in q_t) else v_t[j]
        mujoco.mj_step(model, data)
        if args.motors == "implicit":
            nsat += int(sum(abs(data.actuator_force[4 + w]) >= 0.95 * WHEEL_TAU for w in range(2)))
    q_prev, v_prev = q_new, v_new
    sat.append(nsat / (2 * SUB))
    if args.video:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
    t = (k + 1) * DT * SUB
    p = pitch_deg(data.qpos[3:7])
    disp = float(np.linalg.norm(data.qpos[0:2] - xy0))
    rows.append(dict(time_s=t, pitch_deg=p, displacement_m=disp, height_m=float(data.qpos[2]),
                     speed_m_s=float(np.linalg.norm(data.qvel[0:2])),
                     clear_L_m=float(data.xpos[wheel_bodies[0]][2] - WHEEL_R), clear_R_m=float(data.xpos[wheel_bodies[1]][2] - WHEEL_R)))
    R = quat_to_mat(data.qpos[3:7])
    up = -model.opt.gravity / np.linalg.norm(model.opt.gravity)
    if math.degrees(math.acos(max(-1.0, min(1.0, float(R[:, 2] @ up))))) > 15.0:
        fell_at = t
        break

pre = [r for r in rows if push_step is None or r["time_s"] < args.push_time]
summary = {
    "checkpoint": str(args.checkpoint), "engine": f"mujoco {mujoco.__version__}", "obs_dim": obs_dim,
    "duration_s": args.duration, "latency_ms": args.latency_ms, "obs_noise": args.obs_noise, "slope_deg": args.slope_deg, "friction": args.friction, "init_pitch_deg": args.init_pitch_deg, "presses": args.presses, "tire": args.tire, "motors": args.motors, "noslip": args.noslip, "dt": DT, "wheel_kd": KD_WHEEL, "solref": args.solref, "push_vel_m_s": args.push_vel, "fell_at_s": fell_at,
    "pre_push_max_pitch_dev_deg": max(abs(r["pitch_deg"] - p0) for r in pre),
    "pre_push_max_drift_m": max(r["displacement_m"] for r in pre),
    "max_displacement_m": max(r["displacement_m"] for r in rows),
    "end_displacement_m": rows[-1]["displacement_m"], "end_speed_m_s": rows[-1]["speed_m_s"],
    "wheel_saturation_frac": float(np.mean(sat)),
    "press_clearance_m": {f"{b}@{t0:g}": [round(max(r[c] for r in rows if t0 <= r["time_s"] < t0 + 1.0), 4) if any(t0 <= r["time_s"] < t0 + 1.0 for r in rows) else None
                                         for c in ("clear_L_m", "clear_R_m")] for b, t0 in PRESSES},
}
summary["pass_no_fall"] = fell_at is None
summary["pass_drift"] = summary["pre_push_max_drift_m"] < 0.05
summary["pass_all"] = summary["pass_no_fall"] and summary["pass_drift"] and (push_step is None or rows[-1]["speed_m_s"] < 0.05)
args.output_dir.mkdir(parents=True, exist_ok=True)
if frames:
    import imageio
    imageio.mimsave(args.output_dir / "mujoco.mp4", frames, fps=50)
(args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
with open(args.output_dir / "trace.csv", "w") as f:
    f.write(",".join(rows[0]) + "\n")
    for r in rows:
        f.write(",".join(f"{v:.5f}" for v in r.values()) + "\n")
print(json.dumps(summary, indent=2))
