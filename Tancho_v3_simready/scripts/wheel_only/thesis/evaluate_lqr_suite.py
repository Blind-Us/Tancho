#!/usr/bin/env python3
"""(Copied from outputs/rl_vs_lqr_isaac_20261102/, + --obs-noise / --latency-substeps.)
Run the Stability RL policy through the report's LQR test suite (sections 4.2-4.5) in Isaac Sim.

One launch, one env per case, all cases in parallel.  Nominal Play physics (no noise,
no randomisation, friction 0.8x0.8), 200 Hz physics / 50 Hz policy.  Termination terms
are removed; failure is |pitch - balance| >= 15 deg or > 10 N contact on base/thigh/calf
(same definition as the LQR tests).  Scenarios:
  pitch   initial pitch = balance + {±1,±3,±5,±7,±10} deg, zero command, 5 s
  cmd     vx 0.3 m/s on t in [1,5); yaw 1.0 rad/s on [1,4); 0.2 m/s + 0.5 rad/s on [1,5)
  push    50 ms horizontal force on base_link_root COM along +heading at t = 2 s, 0-100 N
  rate    +5 / +10 deg initial pitch, policy action held for 1..5 policy steps (50..10 Hz)
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Stability-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--latency-substeps", type=int, default=0, help="deployment check: hold the previous motor targets for this many 5 ms substeps")
parser.add_argument("--obs-noise", action="store_true", help="deployment check: keep the training sensor noise")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.scene import LEG_JOINTS, WHEEL_JOINTS  # noqa: E402

PUSH_T, PULSE = 2.0, 0.05
FORCES = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 36, 38, 39, 40, 42, 45, 50, 55, 60, 70, 80, 90, 100]
CASES = []
for off in [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]:
    CASES.append(dict(kind="pitch", name=f"pitch{off:+d}", pitch=off))
CASES += [dict(kind="cmd", name="vx0.3", vx=0.3, wz=0.0, t0=1.0, t1=5.0),
          dict(kind="cmd", name="yaw1.0", vx=0.0, wz=1.0, t0=1.0, t1=4.0),
          dict(kind="cmd", name="vx0.2_yaw0.5", vx=0.2, wz=0.5, t0=1.0, t1=5.0)]
CASES += [dict(kind="push", name=f"push{f}N", force=float(f)) for f in FORCES]
for p in (5, 10):
    for hold in (1, 2, 3, 4, 5):
        CASES.append(dict(kind="rate", name=f"rate{50/hold:.1f}Hz_p{p}", pitch=p, hold=hold))
N = len(CASES)


def pitch_of(q):  # +nose-down rotation about world y (wxyz)
    w, x, y, z = q.unbind(-1)
    return torch.rad2deg(torch.asin(torch.clamp(2 * (w * y - z * x), -1, 1)))


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def qrot(q, v):
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], -1)
    qc = q * torch.tensor([1, -1, -1, -1], device=q.device)
    return qmul(qmul(q, qv), qc)[..., 1:]


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=N, use_fabric=True)
    cfg.seed = 42
    cfg.episode_length_s = args.duration + 5.0
    cfg.sim.physx.enable_external_forces_every_iteration = True
    cfg.curriculum = None
    cfg.events.push_robot = None
    if args.latency_substeps:
        for term in (cfg.actions.leg_pos, cfg.actions.wheel_vel):
            term.min_delay_substeps = term.max_delay_substeps = args.latency_substeps
    if args.obs_noise:
        cfg.observations.policy.enable_corruption = True
    for t in list(cfg.terminations.to_dict()):
        setattr(cfg.terminations, t, None)
    c = cfg.commands.base_velocity
    c.rel_standing_envs = 0.0; c.resampling_time_range = (1e6, 1e6)
    c.ranges.lin_vel_x = c.ranges.lin_vel_y = c.ranges.ang_vel_z = (0.0, 0.0)
    env = gym.make(args.task, cfg=cfg); core = env.unwrapped; dev = core.device
    robot = core.scene["robot"]; contact = core.scene["contact_forces"]
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=dev).eval()
    wheel_ids, _ = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    leg_ids, _ = robot.find_joints(LEG_JOINTS, preserve_order=True)
    wheel_b, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    root_b, _ = robot.find_bodies(["base_link_root"])
    bad_b, bad_names = contact.find_bodies(["base_link_root", "thigh_.*", "calf_.*"])
    cmd_term = core.command_manager.get_term("base_velocity")
    dt = core.step_dt; steps = int(round(args.duration / dt))

    obs, _ = env.reset()
    # balance pitch: whole-robot COM straight above the axle midpoint
    mass = robot.data.default_mass.to(dev)
    com = (robot.data.body_com_pos_w * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
    axle = robot.data.body_link_pos_w[:, wheel_b].mean(1)
    rel = com - axle
    p_now = pitch_of(robot.data.root_link_quat_w)
    balance = p_now - torch.rad2deg(torch.atan2(rel[:, 0], rel[:, 2]))
    # initial pitch per case (rotate the whole robot about the axle)
    p0 = torch.tensor([cs.get("pitch", 0.0) for cs in CASES], device=dev, dtype=torch.float) + balance
    th = torch.deg2rad(p0 - p_now)
    qy = torch.stack([torch.cos(th / 2), torch.zeros_like(th), torch.sin(th / 2), torch.zeros_like(th)], -1)
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :3] = axle + qrot(qy, pose[:, :3] - axle)
    pose[:, 3:7] = qmul(qy, pose[:, 3:7])
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    core.scene.write_data_to_sim(); core.sim.forward(); core.scene.update(dt=core.physics_dt)
    obs = core.observation_manager.compute()
    xy0 = robot.data.root_link_pos_w[:, :2].clone(); yaw0 = yaw_of(robot.data.root_link_quat_w)
    heading = torch.stack([torch.cos(yaw0), torch.sin(yaw0), torch.zeros_like(yaw0)], -1)

    kind = [cs["kind"] for cs in CASES]
    force = torch.tensor([cs.get("force", 0.0) for cs in CASES], device=dev)
    hold = torch.tensor([cs.get("hold", 1) for cs in CASES], device=dev)
    action = torch.zeros(N, 6, device=dev)
    yaw_acc = torch.zeros(N, device=dev); prev_yaw = yaw0.clone()
    failed_at = [None] * N; contact_at = [None] * N
    rows = [[] for _ in range(N)]

    def record(t, F):
        q = robot.data.root_link_quat_w
        p = pitch_of(q)
        vb = qrot(q * torch.tensor([1, -1, -1, -1], device=dev), robot.data.root_lin_vel_w)
        tau = robot.data.applied_torque[:, wheel_ids]
        disp = ((robot.data.root_link_pos_w[:, :2] - xy0) * heading[:, :2]).sum(1)
        cmd = cmd_term.vel_command_b
        vals = torch.stack([p - balance, disp, torch.linalg.norm(robot.data.root_link_pos_w[:, :2] - xy0, dim=1), vb[:, 0],
                            robot.data.root_ang_vel_w[:, 2], yaw_acc, tau[:, 0], tau[:, 1], cmd[:, 0], cmd[:, 2], F], 1).cpu().tolist()
        legs = robot.data.joint_pos[:, leg_ids].cpu().tolist()
        for i in range(N):
            r = dict(zip(["pitch_dev_deg", "x_m", "disp_m", "vx_body_m_s", "yaw_rate_rad_s", "yaw_rad", "tau_L_Nm", "tau_R_Nm", "cmd_vx", "cmd_wz", "force_N"], vals[i]))
            r["time_s"] = round(t, 4); r.update(dict(zip(["thigh_L", "calf_L", "thigh_R", "calf_R"], legs[i])))
            rows[i].append(r)

    record(0.0, torch.zeros(N, device=dev))
    for k in range(steps):
        t = k * dt
        # commands
        cmd = torch.zeros(N, 3, device=dev)
        for i, cs in enumerate(CASES):
            if cs["kind"] == "cmd" and cs["t0"] <= t + 1e-9 < cs["t1"]:
                cmd[i, 0] = cs["vx"]; cmd[i, 2] = cs["wz"]
        cmd_term.vel_command_b[:] = cmd
        obs = core.observation_manager.compute()  # refresh command in the policy input
        # push: fraction of this policy step overlapping [PUSH_T, PUSH_T + PULSE]
        ov = max(0.0, min(t + dt, PUSH_T + PULSE) - max(t, PUSH_T)) / dt
        F = force * ov
        fw = F[:, None] * heading
        fl = qrot(robot.data.root_link_quat_w * torch.tensor([1, -1, -1, -1], device=dev), fw)[:, None, :].contiguous()
        robot.permanent_wrench_composer.set_forces_and_torques(forces=fl, torques=torch.zeros_like(fl),
            body_ids=torch.tensor(root_b, dtype=torch.int32, device=dev), is_global=False)
        with torch.inference_mode():
            a = policy(obs["policy"])
        upd = (k % hold) == 0
        action = torch.where(upd[:, None], a, action)
        obs, _, _, _, _ = env.step(action)
        y = yaw_of(robot.data.root_link_quat_w); yaw_acc += torch.remainder(y - prev_yaw + math.pi, 2 * math.pi) - math.pi; prev_yaw = y
        pd = (pitch_of(robot.data.root_link_quat_w) - balance).abs().cpu().tolist()
        cf = contact.data.net_forces_w_history[:, :, bad_b].norm(dim=-1).amax(dim=(1, 2)).cpu().tolist()
        for i in range(N):
            if failed_at[i] is None and pd[i] >= 15.0: failed_at[i] = round(t + dt, 4)
            if contact_at[i] is None and cf[i] > 10.0: contact_at[i] = round(t + dt, 4)
        record(t + dt, F)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "timeseries.csv").open("w", newline="") as s:
        w = csv.DictWriter(s, fieldnames=["case"] + list(rows[0][0])); w.writeheader()
        for i, cs in enumerate(CASES):
            for r in rows[i]: w.writerow({"case": cs["name"], **r})
    meta = dict(checkpoint=str(args.checkpoint.resolve()), task=args.task, policy_dt_s=dt, physics_dt_s=core.physics_dt,
                balance_pitch_deg=float(balance.mean()), push_time_s=PUSH_T, pulse_s=PULSE, duration_s=args.duration,
                contact_bodies=bad_names, cases=CASES, failed_at=dict(zip([c["name"] for c in CASES], failed_at)),
                contact_at=dict(zip([c["name"] for c in CASES], contact_at)),
                note="pitch_dev_deg is relative to the geometric balance pitch (COM over axle); torques are PhysX applied wheel torques sampled at 50 Hz")
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("DONE", json.dumps(meta["failed_at"]), json.dumps(meta["contact_at"]))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
