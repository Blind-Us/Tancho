"""Project-local command terms used by the staged Tancho tasks."""

from __future__ import annotations

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class LowArrowVelocityCommand(UniformVelocityCommand):
    """Draw velocity arrows close enough to the chassis to remain in view."""

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.20
        desired_scale, desired_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        actual_scale, actual_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, desired_quat, desired_scale)
        self.current_vel_visualizer.visualize(base_pos_w, actual_quat, actual_scale)
