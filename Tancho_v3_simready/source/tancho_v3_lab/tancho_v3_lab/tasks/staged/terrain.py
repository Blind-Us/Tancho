"""Terrains and the terrain curriculum for the 6-DOF walk stages beyond flat ground.

Scale reference: tire radius 36.1 mm, axle track 0.23 m.

* Rough (``TanchoV3-Walk-Rough``): random bumps whose amplitude grows with the
  curriculum level, gentle pyramid slopes (up and down) and a flat share so the
  flat skill is kept.
* Step (``TanchoV3-Walk-Step``): pyramid stairs, normal (step down off the
  center platform) and inverted (step up), with step height 0 -> 3 cm over the
  levels.  Wide treads (0.6 m) so each edge is crossed on its own.

Curriculum: the built-in ``terrain_levels_vel`` measures straight-line distance
from the spawn point, which does not suit yaw-rate commands (a 0.6 m/s,
1 rad/s command drives a 0.6 m circle).  ``terrain_levels_tracking`` instead
promotes a robot that survived the whole episode while tracking the commanded
speed, and demotes one that fell.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import torch
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.utils import configclass

from .scene import NOMINAL_FRICTION

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -- rough bumps whose amplitude follows the difficulty -----------------------------
def scaled_random_uniform_terrain(difficulty: float, cfg: "ScaledRandomUniformTerrainCfg"):
    """``random_uniform_terrain`` with noise_range = difficulty-interpolated amplitude (the stock one ignores difficulty)."""
    amp = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    amp = max(amp, cfg.noise_step)
    scaled = copy.copy(cfg)
    scaled.noise_range = (-0.5 * amp, 0.5 * amp)
    return random_uniform_terrain(difficulty, scaled)


@configclass
class ScaledRandomUniformTerrainCfg(terrain_gen.HfRandomUniformTerrainCfg):
    function = scaled_random_uniform_terrain
    amplitude_range: tuple[float, float] = (0.0, 0.02)
    """Peak-to-peak bump height (m) at difficulty 0 and 1."""


_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=NOMINAL_FRICTION,
    dynamic_friction=NOMINAL_FRICTION,
    restitution=0.0,
)

ROUGH_GENERATOR = TerrainGeneratorCfg(
    seed=0,
    size=(6.0, 6.0),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=None,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        # 0 -> 2 cm peak-to-peak bumps on a 10 cm grid (smooth spline in between).
        "rough": ScaledRandomUniformTerrainCfg(
            proportion=0.45, amplitude_range=(0.0, 0.02), noise_range=(0.0, 0.0), noise_step=0.001,
            downsampled_scale=0.1, border_width=0.25,
        ),
        # 0 -> ~8 deg (slope 0.14), robot spawns on the top / bottom platform.
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.14), platform_width=1.5, border_width=0.25
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.14), platform_width=1.5, border_width=0.25
        ),
    },
)

STEP_GENERATOR = TerrainGeneratorCfg(
    seed=0,
    size=(6.0, 6.0),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=None,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # Keep some rough so the previous skill is not forgotten.
        "rough": ScaledRandomUniformTerrainCfg(
            proportion=0.2, amplitude_range=(0.0, 0.02), noise_range=(0.0, 0.0), noise_step=0.001,
            downsampled_scale=0.1, border_width=0.25,
        ),
        # Spawn on the top platform: every edge is a step down.
        "step_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.35, step_height_range=(0.005, 0.03), step_width=0.6, platform_width=1.5, border_width=0.3
        ),
        # Spawn in the pit: every edge is a step up.
        "step_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35, step_height_range=(0.005, 0.03), step_width=0.6, platform_width=1.5, border_width=0.3
        ),
    },
)

CLIMB_GENERATOR = TerrainGeneratorCfg(
    seed=0,
    size=(6.0, 6.0),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=None,
    use_cache=False,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "rough": ScaledRandomUniformTerrainCfg(
            proportion=0.15, amplitude_range=(0.0, 0.02), noise_range=(0.0, 0.0), noise_step=0.001,
            downsampled_scale=0.1, border_width=0.25,
        ),
        # Step up is the new skill: 1 -> 3 cm over the levels.
        "step_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5, step_height_range=(0.01, 0.03), step_width=0.6, platform_width=1.5, border_width=0.3
        ),
        "step_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.01, 0.03), step_width=0.6, platform_width=1.5, border_width=0.3
        ),
    },
)


def make_terrain(generator: TerrainGeneratorCfg, max_init_level: int | None = 2) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=max_init_level,
        collision_group=-1,
        physics_material=_MATERIAL,
        debug_vis=False,
    )


def play_generator(generator: TerrainGeneratorCfg, sub_terrain: str, difficulty: float, size: float = 16.0) -> TerrainGeneratorCfg:
    """One large tile of a single sub-terrain at a fixed difficulty (evaluation / video)."""
    gen = copy.deepcopy(generator)
    gen.sub_terrains = {sub_terrain: gen.sub_terrains[sub_terrain]}
    gen.sub_terrains[sub_terrain].proportion = 1.0
    gen.size = (size, size)
    gen.num_rows = 1
    gen.num_cols = 1
    gen.curriculum = False
    gen.difficulty_range = (difficulty, difficulty)
    return gen


# -- curriculum --------------------------------------------------------------------
def terrain_levels_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term: str = "lin_vel_xy",
    promote_ratio: float = 0.6,
) -> torch.Tensor:
    """Promote: survived to time-out and kept ``promote_ratio`` of the maximum
    ``reward_term`` (exp tracking kernel, 1.0 per second when perfect).
    Demote: terminated by a failure (tilt / body contact).

    Runs before the reward manager resets, so the episode sums are still valid.
    """
    terrain = env.scene.terrain
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())
    term_cfg = env.reward_manager.get_term_cfg(reward_term)
    episode_s = env.episode_length_buf[env_ids_t].float() * env.step_dt
    best = term_cfg.weight * episode_s.clamp(min=env.step_dt)
    ratio = env.reward_manager._episode_sums[reward_term][env_ids_t] / best
    timed_out = env.termination_manager.time_outs[env_ids_t]
    failed = env.termination_manager.terminated[env_ids_t]
    move_up = timed_out & (ratio > promote_ratio)
    move_down = failed & ~move_up
    terrain.update_env_origins(env_ids_t, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
