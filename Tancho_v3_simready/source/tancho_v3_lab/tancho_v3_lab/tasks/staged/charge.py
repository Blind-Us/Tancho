"""Charged hop (``TanchoV3-HopCharge``, user 2026-11-02).

The user liked the standard hop (round 3, tag ``hop-push-r3-stable``) and asked for a
second hop in the same policy: hold a button to crouch and charge, release to jump
higher.  The standard hop stays on LT+RT (teleop: Shift); the charge is a separate
held button (teleop: J), command ``charge`` (0/1).

* Held: both legs go down the vertical line through the nominal axle (URDF FK, root
  frame x = -16.5 mm for dthigh = -0.41 * dcalf) to axle -172 mm (nominal -224) in
  ``CROUCH_S``; the crouch depth reached (``level``, 0-1) is the charge.
* Released: step targets to the -245 mm extension (held 0.06 s), then a tuck that
  uses the thigh as much as the knee (the user: earlier hops tucked with the calf
  only, avoid that), then back to nominal.

Policy observation +3: held, level, time since release (0 -> 1 over 0.6 s, 1 idle).
While charging and for ``recent_s`` after release the climb term's posture gates
(``recently_pressed``) are held open, so the climb hop's relaxed costs / tilt limit
apply to the charged hop too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .climb import _ground_under, _keyframes, _wheel_scan
from .scene import FULL_ROOT_HEIGHT_M, WHEEL_RADIUS_M

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CROUCH = (-0.305, 0.75)
CROUCH_S = 0.35
EXTEND = (0.245, -0.60)
TUCK = (-0.305, 0.75)  # thigh -0.305 rad: the axle comes up under the body, not behind it
CHARGE_STEP_S = 0.01
# (t after release, dthigh, dcalf): extension held, tuck, hold, return.
CHARGE_KEYS = ((CHARGE_STEP_S, *EXTEND), (0.07, *EXTEND), (0.08, *TUCK), (0.16, *TUCK), (0.28, 0.0, 0.0))
CHARGE_END_S = CHARGE_KEYS[-1][0]
CHARGE_PHASE_CLIP_S = 0.6


class ChargeCommand(CommandTerm):
    cfg: "ChargeCommandCfg"

    def __init__(self, cfg: "ChargeCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.held = torch.zeros(n, device=dev)
        self.level = torch.zeros(n, device=dev)
        self.release_level = torch.zeros(n, device=dev)
        self.phase = torch.full((n,), 10.0, device=dev)  # s since release
        self.hold_timer = torch.zeros(n, device=dev)
        self.metrics["charge_s"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self.held.unsqueeze(1)

    @property
    def obs(self) -> torch.Tensor:
        return torch.stack([self.held, self.level, self.phase.clamp(max=CHARGE_PHASE_CLIP_S) / CHARGE_PHASE_CLIP_S], dim=1)

    @property
    def active(self) -> torch.Tensor:
        """Charging, or the released hop still running."""
        return (self.held > 0.0) | (self.phase < CHARGE_END_S)

    def _update_metrics(self):
        self.metrics["charge_s"] += self.held * self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = len(ids)
        idle = (self.hold_timer[ids] <= 0.0) & (self.phase[ids] > 1.0)
        start = idle & (torch.rand(k, device=self.device) < self.cfg.press_prob)
        dur = torch.empty(k, device=self.device).uniform_(*self.cfg.press_s)
        self.hold_timer[ids] = torch.where(start, dur, self.hold_timer[ids])

    def _update_command(self):
        dt = self._env.step_dt
        new = (self.hold_timer > 0.0).float()
        self.hold_timer = (self.hold_timer - dt).clamp(min=0.0)
        released = (new <= 0.0) & (self.held > 0.0)
        self.release_level = torch.where(released, self.level, self.release_level)
        self.phase = torch.where(released, torch.zeros_like(self.phase), self.phase + dt)
        self.level = torch.where(new > 0.0, (self.level + dt / CROUCH_S).clamp(max=1.0), torch.zeros_like(self.level))
        self.held = new
        climb = self._env.command_manager.get_term(self.cfg.climb_command_name)
        climb.recent_timer = torch.where(self.active, torch.full_like(climb.recent_timer, climb.cfg.recent_s), climb.recent_timer)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.held[ids] = 0.0
        self.level[ids] = 0.0
        self.release_level[ids] = 0.0
        self.phase[ids] = 10.0
        self.hold_timer[ids] = 0.0
        return super().reset(env_ids)


@configclass
class ChargeCommandCfg(CommandTermCfg):
    class_type: type = ChargeCommand
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    climb_command_name: str = "climb"
    press_prob: float = 0.5
    press_s: tuple[float, float] = (0.15, 1.5)
    """Hold time (s): short presses are shallow charges, >= ``CROUCH_S`` full ones."""
    jump: bool = True
    """False: releasing stands back up (squat only), no push-off."""


def charge_offsets(term: ChargeCommand) -> torch.Tensor:
    """(N, 4) leg offsets thigh_L, calf_L, thigh_R, calf_R (both legs the same)."""
    dev = term.device
    crouch = torch.tensor(CROUCH, device=dev)
    held = term.held.unsqueeze(1) * term.level.unsqueeze(1) * crouch  # (N, 2)
    p = term.phase.unsqueeze(1)  # (N, 1)
    start = term.release_level.unsqueeze(1) * crouch
    if not term.cfg.jump:
        # Squat only: back up linearly from the crouch reached, over the hop's duration.
        rise = start * (1.0 - p / CHARGE_END_S).clamp(0.0, 1.0)
        off = torch.where(term.held.unsqueeze(1) > 0.0, held, torch.where(p < CHARGE_END_S, rise, torch.zeros_like(rise)))
        return off.repeat(1, 2)
    w = (p / CHARGE_STEP_S).clamp(0.0, 1.0)
    first = start * (1.0 - w) + torch.tensor(EXTEND, device=dev) * w
    rest = _keyframes(p, CHARGE_KEYS).squeeze(1)  # (N, 2); 0 before the first / after the last key
    after = torch.where(p < CHARGE_STEP_S, first, rest)
    off = torch.where(term.held.unsqueeze(1) > 0.0, held, torch.where(p < CHARGE_END_S, after, torch.zeros_like(after)))
    return off.repeat(1, 2)


def charge_obs(env: ManagerBasedRLEnv, command_name: str = "charge") -> torch.Tensor:
    return env.command_manager.get_term(command_name).obs


def charge_tracking(env: ManagerBasedRLEnv, command_name: str = "charge", std: float = 0.25) -> torch.Tensor:
    """While charging / hopping: per leg exp(-|q - q_ref|^2 / std^2), summed over both legs."""
    term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    if not hasattr(env, "_climb_leg_ids"):
        env._climb_leg_ids = robot.find_joints(["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"], preserve_order=True)[0]
    q = robot.data.joint_pos[:, env._climb_leg_ids] - robot.data.default_joint_pos[:, env._climb_leg_ids]
    err = (q - charge_offsets(term)).square().reshape(-1, 2, 2).sum(-1)
    return term.active.float() * torch.exp(-err / std**2).sum(dim=1)


def charge_flight_rise(env: ManagerBasedRLEnv, command_name: str = "charge", sensor_name: str = "height_scanner",
                       min_clearance: float = 0.005, max_rise: float = 0.08) -> torch.Tensor:
    """After a release: root height above standing while both tires are off the ground, [0, 1] at ``max_rise``."""
    term = env.command_manager.get_term(command_name)
    dx, dy, hit_z, wheel_z = _wheel_scan(env, sensor_name)
    ground = _ground_under(dx, dy, hit_z)
    airborne = ((wheel_z - WHEEL_RADIUS_M - ground) > min_clearance).all(dim=1)
    rise = env.scene["robot"].data.root_pos_w[:, 2] - ground.mean(dim=1) - FULL_ROOT_HEIGHT_M
    flying = term.phase < 0.4
    return (flying & airborne).float() * rise.clamp(0.0, max_rise) / max_rise
