#!/usr/bin/env python3
"""Drive the 6-DOF walk policy with the keyboard (``TanchoV3-Walk-Flat-Play-v0``).

Click the Isaac Sim viewport first so it has keyboard focus, then:

  W / S      forward / backward   (hold)
  A / D      turn left / right    (hold)
  1 / 2 / 3  speed level: slow / medium / fast
  R          reset the robot

Speed levels are inside the training ranges (vx +/-0.6 m/s, yaw rate +/-1.0 rad/s).
Releasing every key commands zero, i.e. stand still.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="TanchoV3-Walk-Flat-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True, help="exported TorchScript policy.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import carb  # noqa: E402
import gymnasium as gym  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import tancho_v3_lab.tasks  # noqa: E402,F401

# (vx m/s, yaw rate rad/s) per speed level
SPEED_LEVELS = {"KEY_1": (0.2, 0.5), "KEY_2": (0.4, 0.8), "KEY_3": (0.6, 1.0)}
MOVE_KEYS = {"W": (1, 0), "S": (-1, 0), "A": (0, 1), "D": (0, -1)}


class WasdKeyboard:
    def __init__(self):
        self.held: set[str] = set()
        self.vx_max, self.wz_max = SPEED_LEVELS["KEY_2"]
        self.reset_requested = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args, **kwargs) -> bool:
        # CHAR events carry the typed character as a plain str; only key press/release matter here.
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
            return True
        name = event.input if isinstance(event.input, str) else event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name in MOVE_KEYS:
                self.held.add(name)
            elif name in SPEED_LEVELS:
                self.vx_max, self.wz_max = SPEED_LEVELS[name]
                print(f"[teleop] speed level: vx {self.vx_max} m/s, yaw {self.wz_max} rad/s", flush=True)
            elif name == "R":
                self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(name)
        return True

    def command(self) -> tuple[float, float]:
        fwd = sum(MOVE_KEYS[k][0] for k in self.held)
        turn = sum(MOVE_KEYS[k][1] for k in self.held)
        return fwd * self.vx_max, turn * self.wz_max


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)
    cfg.episode_length_s = 1.0e6  # no time-out reset while driving
    cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = ranges.lin_vel_y = ranges.ang_vel_z = (0.0, 0.0)
    env = gym.make(args.task, cfg=cfg)
    core = env.unwrapped
    command = core.command_manager.get_term("base_velocity")
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location=core.device).eval()
    keys = WasdKeyboard()
    print(__doc__, flush=True)

    obs, _ = env.reset()
    while simulation_app.is_running():
        start = time.time()
        if keys.reset_requested:
            keys.reset_requested = False
            obs, _ = env.reset()
        vx, wz = keys.command()
        command.vel_command_b[0] = torch.tensor([vx, 0.0, wz], device=core.device)
        with torch.inference_mode():
            obs, _, term, _, _ = env.step(policy(obs["policy"]))
        if bool(term[0]):
            print("[teleop] fell over (tilt > 15 deg or body contact); reset", flush=True)
        # Real time: one 0.02 s policy step per 0.02 s of wall clock.
        time.sleep(max(0.0, core.step_dt - (time.time() - start)))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
