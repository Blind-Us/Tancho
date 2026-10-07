"""Operator-triggered step climbing (``TanchoV3-Climb``).

On the robot the operator holds an Xbox controller: LT lifts the left leg, RT
lifts the right leg, LT+RT together is a short hop.  The policy sees the two
triggers as a 2-dim command ``climb`` (each 0 or 1, analog trigger > 0.5).

A blind policy cannot know where a step is; the operator does.  In training a
simulated operator presses the triggers:

* Auto (the useful presses): a privileged height scan in front of the robot
  (never an actor input) finds a rise > ``rise_threshold`` ahead of a wheel
  while the robot is commanded forward.  When the tire will reach the edge
  within a per-env random time (0.1-0.3 s: an early or late human), the
  trigger on that wheel's side is pressed and held for ``hold_s`` (one press per edge, then ``cooldown_s``).  A square approach shows the edge to both wheels at once -> hop;
  a diagonal approach reaches one wheel first -> single-leg lift.
* Random: occasional presses (L, R or both, 0.2-0.5 s) anywhere, so a press on
  flat ground must not cause a fall.
* ``auto_prob``: per resample, the share of time the operator is attentive.

Reward terms here: wheel clearance on the pressed side, and gated versions of
the posture terms (mirror, leg deviation, vertical velocity) that would
otherwise forbid lifting a leg.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.envs.mdp as mdp
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from ..direct.tancho_v3 import custom_rewards as cr
from .scene import FULL_ROOT_HEIGHT_M, WHEEL_RADIUS_M

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

WHEEL_BODIES = ["wheel_L", "wheel_R"]  # command[:, 0] = LT = left, command[:, 1] = RT = right

# -- reference lift (from scripts/wheel_only/leg_lift_feasibility.py) -----------------
# Leg offsets from the nominal pose (dthigh, dcalf), started on the trigger's rising
# edge: a short push (axle 21 mm away from the body), a tuck (axle 42 mm toward the
# body), then a linear return.  Both poses keep the axle within 1 cm of its nominal
# horizontal position.  The best scripted variant lifted the tire 26.6 mm on flat
# ground but always fell (pitch reaction of the thigh swing) - the policy must add
# the balance.
REF_EXTEND = (0.275, -0.6)
REF_TUCK = (-0.3, 0.6)
REF_T_PUSH = 0.02
REF_T_TUCK_END = 0.22
REF_T_END = 0.40
PHASE_CLIP_S = 0.6


def reference_leg_offsets(phase: torch.Tensor) -> torch.Tensor:
    """(N, 2) time since each side's press -> (N, 4) offsets thigh_L, calf_L, thigh_R, calf_R."""
    ext = torch.tensor(REF_EXTEND, device=phase.device)
    tuck = torch.tensor(REF_TUCK, device=phase.device)
    p = phase.unsqueeze(-1)  # (N, 2, 1)
    back = (1.0 - (p - REF_T_TUCK_END) / (REF_T_END - REF_T_TUCK_END)).clamp(0.0, 1.0)
    off = torch.where(p < REF_T_PUSH, ext, torch.where(p < REF_T_TUCK_END, tuck, tuck * back))
    off = torch.where(p < REF_T_END, off, torch.zeros_like(off))
    return off.reshape(-1, 4)


def _wheel_scan(env: ManagerBasedRLEnv, sensor_name: str, robot_name: str = "robot"):
    """Per wheel: local (yaw-frame) ray offsets from the wheel and ground heights.

    Returns ``dx`` (N, 2, R), ``dy`` (N, 2, R), ``hit_z`` (N, R), ``wheel_z`` (N, 2).
    """
    sensor = env.scene.sensors[sensor_name]
    robot = env.scene[robot_name]
    if not hasattr(env, "_climb_wheel_ids"):
        env._climb_wheel_ids = robot.find_bodies(WHEEL_BODIES, preserve_order=True)[0]
    wheel_w = robot.data.body_pos_w[:, env._climb_wheel_ids]  # (N, 2, 3)
    q = yaw_quat(robot.data.root_quat_w)
    hits = sensor.data.ray_hits_w  # (N, R, 3)
    rel = hits.unsqueeze(1) - wheel_w.unsqueeze(2)  # (N, 2, R, 3)
    n, _, r, _ = rel.shape
    rel_b = quat_apply_inverse(q.repeat_interleave(2 * r, dim=0), rel.reshape(-1, 3)).reshape(n, 2, r, 3)
    hit_z = torch.nan_to_num(hits[..., 2], nan=-10.0, posinf=-10.0, neginf=-10.0)
    return rel_b[..., 0], rel_b[..., 1], hit_z, wheel_w[..., 2]


def _ground_under(dx, dy, hit_z):
    """Ground height under each wheel: highest hit within 4 cm of the contact point."""
    near = (dx.abs() < 0.04) & (dy.abs() < 0.04)
    z = torch.where(near, hit_z.unsqueeze(1).expand_as(dx), torch.full_like(dx, -10.0))
    z_max = z.max(dim=-1).values
    # Fallback (no ray close enough): nearest ray.
    nearest = (dx.square() + dy.square()).argmin(dim=-1, keepdim=True)
    z_near = torch.gather(hit_z.unsqueeze(1).expand_as(dx), -1, nearest).squeeze(-1)
    return torch.where(z_max > -9.0, z_max, z_near)


class ClimbTriggerCommand(CommandTerm):
    cfg: "ClimbTriggerCommandCfg"

    def __init__(self, cfg: "ClimbTriggerCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.trigger = torch.zeros(n, 2, device=dev)
        self.random_mode = torch.zeros(n, 2, device=dev)
        self.random_timer = torch.zeros(n, device=dev)
        self.hold_mode = torch.zeros(n, 2, device=dev)
        self.hold_timer = torch.zeros(n, 2, device=dev)
        self.cooldown = torch.zeros(n, 2, device=dev)
        self.recent_timer = torch.zeros(n, device=dev)
        self.phase = torch.full((n, 2), 10.0, device=dev)  # s since each side's rising edge
        self.lookahead = torch.full((n,), sum(cfg.lookahead_range) / 2, device=dev)
        self.attentive = torch.ones(n, dtype=torch.bool, device=dev)
        self.metrics["press_auto"] = torch.zeros(n, device=dev)
        self.metrics["press_any"] = torch.zeros(n, device=dev)

    @property
    def command(self) -> torch.Tensor:
        return self.trigger

    @property
    def phase_obs(self) -> torch.Tensor:
        """Time since each trigger's rising edge, 0 -> 1 over 0.6 s, 1 when idle (Pi: timer from the press)."""
        return self.phase.clamp(max=PHASE_CLIP_S) / PHASE_CLIP_S

    @property
    def recently_pressed(self) -> torch.Tensor:
        """1 while a trigger is held and for ``recent_s`` after release (lift + landing)."""
        return (self.recent_timer > 0.0).float()

    def _update_metrics(self):
        self.metrics["press_any"] += (self.trigger.max(dim=1).values > 0).float() * self._env.step_dt
        self.metrics["press_auto"] += (self.hold_timer.max(dim=1).values > 0).float() * self._env.step_dt

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        k = len(ids)
        self.lookahead[ids] = torch.empty(k, device=self.device).uniform_(*self.cfg.lookahead_range)
        self.attentive[ids] = torch.rand(k, device=self.device) < self.cfg.auto_prob
        start = torch.rand(k, device=self.device) < self.cfg.random_press_prob
        mode_id = torch.randint(0, 3, (k,), device=self.device)  # 0 L, 1 R, 2 both
        mode = torch.stack([(mode_id != 1).float(), (mode_id != 0).float()], dim=1)
        self.random_mode[ids] = torch.where(start.unsqueeze(1), mode, self.random_mode[ids])
        dur = torch.empty(k, device=self.device).uniform_(*self.cfg.random_press_s)
        self.random_timer[ids] = torch.where(start, dur, torch.zeros_like(dur))

    def _update_command(self):
        dt = self._env.step_dt
        # -- auto (attentive operator)
        dx, dy, hit_z, _ = _wheel_scan(self._env, self.cfg.sensor_name)
        ground = _ground_under(dx, dy, hit_z)  # (N, 2)
        # Nearest ray ahead of each wheel that is a rise; the operator presses when the
        # tire front will reach it within this env's reaction time (time-to-contact),
        # so the 0.4 s lift lands on the edge whatever the speed.
        ahead = (dy.abs() < 0.04) & (dx > 0.0) & (dx < self.cfg.scan_ahead_m)
        is_rise = ahead & ((hit_z.unsqueeze(1) - ground.unsqueeze(-1)) > self.cfg.rise_threshold)
        edge_dx = torch.where(is_rise, dx, torch.full_like(dx, 10.0)).min(dim=-1).values  # (N, 2)
        gap = (edge_dx - WHEEL_RADIUS_M).clamp(min=0.0)
        vx = self._env.scene["robot"].data.root_lin_vel_b[:, 0].clamp(min=0.05)
        ttc = gap / vx.unsqueeze(1)
        vx_cmd = self._env.command_manager.get_command(self.cfg.velocity_command_name)[:, 0]
        need = (edge_dx < 9.0) & (ttc < self.lookahead.unsqueeze(1)) & (vx_cmd > 0.05).unsqueeze(1) & self.attentive.unsqueeze(1)
        # One press per edge: hold for hold_s, then a cooldown before the same side can
        # fire again (otherwise a robot parked at an edge keeps the trigger held and
        # could farm the lift rewards).
        fire = need & (self.hold_timer <= 0.0) & (self.cooldown <= 0.0)
        self.hold_timer = torch.where(fire, torch.full_like(self.hold_timer, self.cfg.hold_s), (self.hold_timer - dt).clamp(min=0.0))
        self.cooldown = torch.where(fire, torch.full_like(self.cooldown, self.cfg.hold_s + self.cfg.cooldown_s), (self.cooldown - dt).clamp(min=0.0))
        auto = (self.hold_timer > 0.0).float()
        # -- random presses
        rnd = self.random_mode * (self.random_timer > 0.0).float().unsqueeze(1)
        self.random_timer = (self.random_timer - dt).clamp(min=0.0)
        new = torch.maximum(auto, rnd)
        rising = (new > 0.0) & (self.trigger <= 0.0)
        self.phase = torch.where(rising, torch.zeros_like(self.phase), self.phase + dt)
        self.trigger = new
        pressed = self.trigger.max(dim=1).values > 0.0
        self.recent_timer = torch.where(pressed, torch.full_like(self.recent_timer, self.cfg.recent_s), (self.recent_timer - dt).clamp(min=0.0))

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.trigger[ids] = 0.0
        self.hold_timer[ids] = 0.0
        self.cooldown[ids] = 0.0
        self.recent_timer[ids] = 0.0
        self.phase[ids] = 10.0
        return super().reset(env_ids)


@configclass
class ClimbTriggerCommandCfg(CommandTermCfg):
    class_type: type = ClimbTriggerCommand
    resampling_time_range: tuple[float, float] = (2.0, 5.0)
    sensor_name: str = "height_scanner"
    velocity_command_name: str = "base_velocity"
    rise_threshold: float = 0.012
    """A rise above this (m) ahead of a wheel counts as a step (rough bumps peak at 2 cm p-p, mostly below)."""
    lookahead_range: tuple[float, float] = (0.1, 0.3)
    """Operator reaction: press when the tire front will reach the edge within this time (s), per env."""
    scan_ahead_m: float = 0.33
    hold_s: float = 0.4
    cooldown_s: float = 0.8
    recent_s: float = 0.6
    auto_prob: float = 0.9
    random_press_prob: float = 0.4
    random_press_s: tuple[float, float] = (0.2, 0.5)


# -- action: legs with optional reference guidance -------------------------------------
class ClimbLegAction(mdp.JointPositionAction):
    """Joint position action + ``scale`` * reference lift offsets (rad).

    The scale is ``env._climb_ref_scale`` when the guidance curriculum sets it,
    else ``cfg.guidance_scale``.  It anneals 1 -> 0 during training so the final
    policy produces the lift itself; Play / deployment use 0 (no Pi-side table)."""

    cfg: "ClimbLegActionCfg"

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        scale = getattr(self._env, "_climb_ref_scale", self.cfg.guidance_scale)
        if scale > 0.0:
            term = self._env.command_manager.get_term(self.cfg.command_name)
            self._processed_actions += scale * reference_leg_offsets(term.phase)


@configclass
class ClimbLegActionCfg(mdp.JointPositionActionCfg):
    class_type: type = ClimbLegAction
    command_name: str = "climb"
    guidance_scale: float = 1.0


def climb_phase(env: ManagerBasedRLEnv, command_name: str = "climb") -> torch.Tensor:
    return env.command_manager.get_term(command_name).phase_obs


def reference_guidance(env: ManagerBasedRLEnv, env_ids, hold_iters: int = 300, anneal_iters: int = 1200, steps_per_iter: int = 24) -> float:
    """Curriculum: reference added to the leg action at full scale for ``hold_iters``, then linearly to 0."""
    it = env.common_step_counter / steps_per_iter
    env._climb_ref_scale = float(min(1.0, max(0.0, 1.0 - (it - hold_iters) / anneal_iters)))
    return env._climb_ref_scale


# -- rewards --------------------------------------------------------------------------
def leg_reference_tracking(env: ManagerBasedRLEnv, command_name: str = "climb", std: float = 0.25) -> torch.Tensor:
    """While a side's reference is running: exp(-|q - q_ref|^2 / std^2) over that leg's two joints."""
    term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    if not hasattr(env, "_climb_leg_ids"):
        env._climb_leg_ids = robot.find_joints(["joint_thigh_L", "joint_calf_L", "joint_thigh_R", "joint_calf_R"], preserve_order=True)[0]
    q = robot.data.joint_pos[:, env._climb_leg_ids] - robot.data.default_joint_pos[:, env._climb_leg_ids]
    err = (q - reference_leg_offsets(term.phase)).square().reshape(-1, 2, 2).sum(-1)
    active = (term.phase < REF_T_END).float()
    return (active * torch.exp(-err / std**2)).sum(dim=1)

def wheel_lift_on_trigger(env: ManagerBasedRLEnv, command_name: str = "climb", sensor_name: str = "height_scanner", max_clearance: float = 0.05) -> torch.Tensor:
    """Sum over pressed sides of tire clearance above the ground under it, normalized to [0, 1] at ``max_clearance``."""
    trig = env.command_manager.get_command(command_name)
    dx, dy, hit_z, wheel_z = _wheel_scan(env, sensor_name)
    clearance = (wheel_z - WHEEL_RADIUS_M - _ground_under(dx, dy, hit_z)).clamp(0.0, max_clearance) / max_clearance
    return (trig * clearance).sum(dim=1)


# Wheel axle below the root (base frame) at the nominal leg pose.
NOMINAL_WHEEL_DROP_M = FULL_ROOT_HEIGHT_M - WHEEL_RADIUS_M


def wheel_retract_on_trigger(env: ManagerBasedRLEnv, command_name: str = "climb", max_retract: float = 0.05) -> torch.Tensor:
    """Dense shaping for the lift: how far the pressed side's axle has been pulled up toward
    the body (base frame) relative to the nominal pose, normalized at ``max_retract``.
    Gives gradient before the tire leaves the ground."""
    trig = env.command_manager.get_command(command_name)
    robot = env.scene["robot"]
    if not hasattr(env, "_climb_wheel_ids"):
        env._climb_wheel_ids = robot.find_bodies(WHEEL_BODIES, preserve_order=True)[0]
    rel = robot.data.body_pos_w[:, env._climb_wheel_ids] - robot.data.root_pos_w.unsqueeze(1)
    q = robot.data.root_quat_w.repeat_interleave(2, dim=0)
    drop = -quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(-1, 2, 3)[..., 2]
    retract = (NOMINAL_WHEEL_DROP_M - drop).clamp(0.0, max_retract) / max_retract
    return (trig * retract).sum(dim=1)


def climb_progress(env: ManagerBasedRLEnv, sensor_name: str = "height_scanner") -> torch.Tensor:
    """Pays for new height: increments of the episode's running maximum of the ground
    height under the wheels (mean of both) above the spawn ground.  Returned as a rate
    (m/s), so the episode sum is weight * best height gained.  Going down, or driving
    back and forth over bumps, earns nothing beyond the first time a height is reached."""
    dx, dy, hit_z, _ = _wheel_scan(env, sensor_name)
    ground = _ground_under(dx, dy, hit_z).mean(dim=1)
    if not hasattr(env, "_climb_ref"):
        env._climb_ref = ground.clone()
        env._climb_best = torch.zeros_like(ground)
        env._climb_init = torch.zeros_like(ground, dtype=torch.bool)
    fresh = (env.episode_length_buf <= 1) | ~env._climb_init
    env._climb_ref = torch.where(fresh, ground, env._climb_ref)
    env._climb_best = torch.where(fresh, torch.zeros_like(ground), env._climb_best)
    env._climb_init |= True
    h = ground - env._climb_ref
    gain = (h - env._climb_best).clamp(min=0.0)
    env._climb_best = torch.maximum(env._climb_best, h)
    return gain / env.step_dt


def _free(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return 1.0 - env.command_manager.get_term(command_name).recently_pressed


def mirror_leg_l2_gated(env: ManagerBasedRLEnv, command_name: str = "climb") -> torch.Tensor:
    return cr.mirror_leg_l2(env) * _free(env, command_name)


def joint_deviation_l1_gated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str = "climb") -> torch.Tensor:
    return mdp.joint_deviation_l1(env, asset_cfg) * _free(env, command_name)


def lin_vel_z_l2_gated(env: ManagerBasedRLEnv, command_name: str = "climb") -> torch.Tensor:
    return mdp.lin_vel_z_l2(env) * _free(env, command_name)
