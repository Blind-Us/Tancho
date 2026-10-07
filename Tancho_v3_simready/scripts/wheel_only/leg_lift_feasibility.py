#!/usr/bin/env python3
"""Can Tancho hop a wheel pair onto a step with legs at nominal +/-0.6 rad?

Scripted test, no learning: the rough-walk policy keeps balance and drives at
``vx`` toward the first edge of an inverted-pyramid step tile (height
``--step-height``; 0 = flat ground, measures pure lift).  When the tire front
is ``gap`` from the edge, the legs are overridden with a fixed hop:

0. crouch (optional pre-load): both legs to the retract pose for ``t_crouch`` s
1. push: both legs to the extend pose (axle 21 mm further from the body) for ``t_push`` s
2. tuck: both legs to the retract pose (axle 42 mm closer to the body) for ``t_tuck`` s
3. back to the nominal pose; the policy has the wheels throughout

Both poses keep the axle within 1 cm (horizontal) of its nominal position under
the body (from the URDF leg kinematics: thigh 0.15 m, calf 0.10 m).

The policy was trained with a 0.25 rad leg scale; it runs in the 0.6 rad
Climb env through a wrapper that rescales its leg outputs and its leg
last-action inputs.  All configurations run in one process (one env, reset
between trials) and are written to ``--output``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True, help="exported 25-input walk/rough policy.pt (0.25 rad legs)")
parser.add_argument("--step-height", type=float, default=0.03)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--vx", type=float, nargs="+", default=[0.3, 0.5])
parser.add_argument("--gap", type=float, nargs="+", default=[0.0, 0.03, 0.06, 0.10])
parser.add_argument("--t-push", type=float, nargs="+", default=[0.0, 0.04, 0.08])
parser.add_argument("--t-crouch", type=float, nargs="+", default=[0.0], help="pre-load: hold the tuck pose before pushing")
parser.add_argument("--leg-kp", type=float, default=None, help="override leg position-loop Kp (hardware: 20)")
parser.add_argument("--leg-kd", type=float, default=None, help="override leg Kd (hardware: 0.2)")
parser.add_argument("--trace", action="store_true", help="print every step after firing (use with one configuration)")
parser.add_argument("--t-tuck", type=float, nargs="+", default=[0.15, 0.25, 0.35])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401
from tancho_v3_lab.tasks.staged.scene import CLIMB_LEG_ACTION_SCALE_RAD, LEG_ACTION_SCALE_RAD, WHEEL_RADIUS_M  # noqa: E402
from tancho_v3_lab.tasks.staged.terrain import CLIMB_GENERATOR, make_terrain, play_generator  # noqa: E402

TASK = "TanchoV3-Climb-Play-v0"
EDGE_X = 0.75  # half the 1.5 m pit platform
# (dthigh, dcalf) offsets from nominal, order thigh_L, calf_L, thigh_R, calf_R.
EXTEND = (0.275, -0.6)
TUCK = (-0.3, 0.6)
K = LEG_ACTION_SCALE_RAD / CLIMB_LEG_ACTION_SCALE_RAD


def leg_action(pose: tuple[float, float], device) -> torch.Tensor:
    t, c = pose
    return torch.tensor([t, c, t, c], device=device) / CLIMB_LEG_ACTION_SCALE_RAD


@torch.inference_mode()
def main() -> int:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = 60.0
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    r = cfg.commands.base_velocity.ranges
    r.lin_vel_x = r.lin_vel_y = r.ang_vel_z = (0.0, 0.0)
    cfg.commands.climb.auto_prob = 0.0  # triggers unused by this policy
    if args.step_height > 0:
        gen = play_generator(CLIMB_GENERATOR, "step_up", 1.0, size=8.0)
        gen.sub_terrains["step_up"].step_height_range = (args.step_height, args.step_height)
    else:
        gen = play_generator(CLIMB_GENERATOR, "flat", 0.0, size=8.0)
    cfg.scene.terrain = make_terrain(gen, max_init_level=None)
    if args.leg_kp is not None:
        cfg.scene.robot.actuators["legs"].stiffness = args.leg_kp
    if args.leg_kd is not None:
        cfg.scene.robot.actuators["legs"].damping = args.leg_kd
    env = gym.make(TASK, cfg=cfg)
    core = env.unwrapped
    robot = core.scene["robot"]
    command = core.command_manager.get_term("base_velocity")
    policy = torch.jit.load(str(args.policy.resolve()), map_location=core.device).eval()
    wheel_ids, _ = robot.find_bodies(["wheel_L", "wheel_R"], preserve_order=True)
    dt = core.step_dt

    def act(obs27: torch.Tensor) -> torch.Tensor:
        o = obs27[:, :25].clone()
        o[:, 11:15] /= K  # new-scale leg last action -> old scale
        a = policy(o)
        a[:, :4] *= K
        return a

    results = []
    for vx, gap, t_crouch, t_push, t_tuck in itertools.product(args.vx, args.gap, args.t_crouch, args.t_push, args.t_tuck):
        obs, _ = env.reset()
        origin = core.scene.env_origins[0]
        ground0 = float(robot.data.body_pos_w[0, wheel_ids, 2].mean()) - WHEEL_RADIUS_M
        phase_t, fired, failed, max_clear, t = None, False, False, 0.0, 0.0
        # Flat: fire after 2 s.  Step: fire when the tire front is `gap` from the edge.
        while t < 12.0:
            command.vel_command_b[0] = torch.tensor([vx, 0.0, 0.0], device=core.device)
            axle = robot.data.body_pos_w[0, wheel_ids]
            axle_x = float(axle[:, 0].mean() - origin[0])
            if not fired and ((args.step_height > 0 and axle_x + WHEEL_RADIUS_M >= EDGE_X - gap) or (args.step_height <= 0 and t >= 2.0)):
                fired, phase_t = True, 0.0
            a = act(obs["policy"])
            if fired and phase_t is not None:
                if phase_t < t_crouch:
                    a[0, :4] = leg_action(TUCK, core.device)
                elif phase_t < t_crouch + t_push:
                    a[0, :4] = leg_action(EXTEND, core.device)
                elif phase_t < t_crouch + t_push + t_tuck:
                    a[0, :4] = leg_action(TUCK, core.device)
                phase_t += dt
            obs, _, term, _, _ = env.step(a)
            t += dt
            if args.trace and fired:
                tm = core.termination_manager
                w, x, y, z = robot.data.root_quat_w[0].tolist()
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))
                q = robot.data.joint_pos[0].tolist()
                print(f"TRACE t={phase_t:.2f} root_z={float(robot.data.root_pos_w[0, 2]):.3f} "
                      f"axle_dz={float(robot.data.body_pos_w[0, wheel_ids, 2].mean()) - WHEEL_RADIUS_M - ground0:+.4f} "
                      f"pitch={pitch:+.1f} q={[round(v, 2) for v in q[:4]]} act={[round(v, 2) for v in a[0].tolist()]} "
                      f"tilt={bool(tm.get_term('tilt')[0])} contact={bool(tm.get_term('body_contact')[0])}", flush=True)
            if bool(term[0]):
                failed = True
                break
            clear = float(robot.data.body_pos_w[0, wheel_ids, 2].min()) - WHEEL_RADIUS_M - ground0
            if fired and phase_t > t_crouch + 0.5 * dt:  # the crouch itself lifts briefly; count from the push
                max_clear = max(max_clear, clear)
            if fired and phase_t is not None and phase_t > t_crouch + t_push + t_tuck + 2.0:
                break
        final_dz = float(robot.data.body_pos_w[0, wheel_ids, 2].mean()) - WHEEL_RADIUS_M - ground0
        climbed = (not failed) and args.step_height > 0 and final_dz > 0.5 * args.step_height
        res = {"vx": vx, "gap": gap, "t_crouch": t_crouch, "t_push": t_push, "t_tuck": t_tuck, "fired": fired, "failed": failed,
               "max_clearance_m": round(max_clear, 4), "final_dz_m": round(final_dz, 4), "climbed": climbed}
        results.append(res)
        print(json.dumps(res), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy": str(args.policy.resolve()),
        "step_height_m": args.step_height,
        "leg_kp": args.leg_kp,
        "leg_kd": args.leg_kd,
        "n": len(results),
        "n_climbed": sum(r["climbed"] for r in results),
        "n_failed": sum(r["failed"] for r in results),
        "max_clearance_m": max(r["max_clearance_m"] for r in results),
        "trials": results,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "trials"}))
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
