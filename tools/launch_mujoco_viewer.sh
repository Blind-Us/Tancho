#!/usr/bin/env bash
set -euo pipefail

# Launch the official empty MuJoCo Simulate GUI.  Models are opened from the
# program UI or by dropping an XML/MJCF file into its window.
exec /home/azul/miniconda3/envs/env_isaaclab/bin/python -m mujoco.viewer
