import torch
from torch import Tensor

from hord.envs.base_env.env_utils.humanoid_utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
)
from hord.envs.base_env.components.base_component import BaseComponent
from hord.envs.base_env.env_utils.general import HistoryBuffer


class HumanoidObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.humanoid_obs = torch.zeros(
            self.env.num_envs,
            self.config.obs_size,
            dtype=torch.float,
            device=self.env.device,
        )
        self.humanoid_obs_hist_buf = HistoryBuffer(
            self.config.num_historical_steps,
            self.env.num_envs,
            shape=(self.config.obs_size,),
            device=self.env.device,
        )

        # Initialize historical_self_obs_with_actions related buffers if enabled
        self.historical_self_obs_with_actions = None
        self.humanoid_action_hist_buf = None
        self.step_count = 0

        if self.config.historical_self_obs_with_actions.enabled:
            # Initialize action history buffer with num_historical_steps (same as obs buffer)
            self.humanoid_action_hist_buf = HistoryBuffer(
                self.config.num_historical_steps,
                self.env.num_envs,
                shape=(self.env.config.robot.number_of_actions,),
                device=self.env.device,
            )
            
            # Store as [num_envs, max_num_historical_steps, obs_per_step]
            obs_per_step = self.config.obs_size + self.env.config.robot.number_of_actions + (1 if self.config.historical_self_obs_with_actions.with_time else 0)
            self.historical_self_obs_with_actions = torch.zeros(
                self.env.num_envs,
                self.config.historical_self_obs_with_actions.max_num_historical_steps,
                obs_per_step,
                dtype=torch.float,
                device=self.env.device,
            )

        body_names = self.env.config.robot.body_names
        num_bodies = len(body_names)
        self.body_contacts = torch.zeros(
            self.env.num_envs,
            num_bodies,
            3,
            dtype=torch.bool,
            device=self.env.device,
        )

    def post_physics_step(self):
        self.humanoid_obs_hist_buf.rotate()
        
        # Only rotate action buffer if enabled
        if self.config.historical_self_obs_with_actions.enabled:
            self.humanoid_action_hist_buf.rotate()
            # Increment step counter, but cap at max history steps
            self.step_count = min(self.step_count + 1, self.config.historical_self_obs_with_actions.max_num_historical_steps)

    def set_current_actions(self, actions, env_ids=None):
        """Set the current actions in the action history buffer"""
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        self.humanoid_action_hist_buf.set_curr(actions, env_ids)

    def reset_envs(self, env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times):
        if self.config.num_historical_steps > 1:
            self.reset_hist_buf(env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times)

        # Reset step counter when environments are reset
        if self.config.historical_self_obs_with_actions.enabled:
            self.step_count = 0

    def _reset_action_history(self, env_ids):
        """Helper method to reset action history with zeros"""
        zero_actions = torch.zeros(
            len(env_ids),
            self.env.config.robot.number_of_actions,
            device=self.env.device
        )
        self.humanoid_action_hist_buf.set_hist(
            zero_actions.unsqueeze(0).expand(self.config.num_historical_steps - 1, -1, -1),
            env_ids
        )

    def reset_hist_buf(self, env_ids, reset_default_env_ids, reset_ref_env_ids, reset_ref_motion_ids, reset_ref_motion_times):
        if len(reset_default_env_ids) > 0:
            self.reset_hist_default(reset_default_env_ids)

        if len(reset_ref_env_ids) > 0:
            self.reset_hist_ref(
                reset_ref_env_ids,
                reset_ref_motion_ids,
                reset_ref_motion_times,
            )

    def reset_hist_default(self, env_ids):
        self.humanoid_obs_hist_buf.set_hist(
            self.humanoid_obs_hist_buf.get_current(env_ids), env_ids=env_ids
        )
        # Only reset action history if enabled
        if self.config.historical_self_obs_with_actions.enabled:
            self._reset_action_history(env_ids)

    def reset_hist_ref(self, env_ids, motion_ids, motion_times):
        dt = self.env.dt
        motion_ids = torch.tile(
            motion_ids.unsqueeze(-1), [1, self.config.num_historical_steps - 1]
        )
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (
            torch.arange(
                0, self.config.num_historical_steps - 1, device=self.env.device
            )
            + 1
        )
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1).clamp(min=0)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        obs_ref = compute_humanoid_observations_max(
            ref_state.rigid_body_pos,
            ref_state.rigid_body_rot,
            ref_state.rigid_body_vel,
            ref_state.rigid_body_ang_vel,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.config.local_root_obs,
            self.config.root_height_obs,
            True,
        )
        self.humanoid_obs_hist_buf.set_hist(
            obs_ref.view(
                len(env_ids), self.config.num_historical_steps - 1, -1
            ).permute(1, 0, 2),
            env_ids,
        )

        # Only reset action history if enabled
        if self.config.historical_self_obs_with_actions.enabled:
            self._reset_action_history(env_ids)

    def compute_observations(self, env_ids):
        current_state = self.env.simulator.get_bodies_state(env_ids)
        body_contacts = self.env.simulator.get_bodies_contact_buf(env_ids)

        ground_heights = self.env.terrain.get_ground_heights(current_state.rigid_body_pos[:, 0]).clone()

        if self.config.use_max_coords_obs:
            obs = compute_humanoid_observations_max(
                current_state.rigid_body_pos,
                current_state.rigid_body_rot,
                current_state.rigid_body_vel,
                current_state.rigid_body_ang_vel,
                ground_heights,
                self.config.local_root_obs,
                self.config.root_height_obs,
                True,
            )

        else:
            dof_state = self.env.simulator.get_dof_state(env_ids)
            dof_pos = dof_state.dof_pos
            dof_vel = dof_state.dof_vel

            root_pos = current_state.rigid_body_pos[:, 0, :]
            root_rot = current_state.rigid_body_rot[:, 0, :]
            root_vel = current_state.rigid_body_vel[:, 0, :]
            root_ang_vel = current_state.rigid_body_ang_vel[:, 0, :]
            key_body_pos = current_state.rigid_body_pos[:, self.env.key_body_ids, :]

            obs = compute_humanoid_observations(
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos,
                dof_vel,
                key_body_pos,
                ground_heights,
                self.config.local_root_obs,
                self.env.simulator.robot_config.dof_obs_size,
                self.env.simulator.get_dof_offsets(),
                self.env.simulator.robot_config.joint_axis,
                True,
            )
        self.body_contacts[:] = body_contacts
        self.humanoid_obs[env_ids] = obs
        self.humanoid_obs_hist_buf.set_curr(obs, env_ids)

        # Compute historical_self_obs_with_actions if enabled
        if self.config.historical_self_obs_with_actions.enabled:
            self.compute_historical_self_obs_with_actions(env_ids)

    def compute_historical_self_obs_with_actions(self, env_ids):
        """Compute historical self observations with actions concatenated"""
        # Get the actual number of steps available
        actual_steps = min(self.step_count, self.config.historical_self_obs_with_actions.max_num_historical_steps)

        # Initialize the sequence with zeros
        self.historical_self_obs_with_actions[env_ids] = 0

        if actual_steps == 0:
            return  # No history available

        # Get historical observations and actions for the actual steps available
        # Note: obs[i] corresponds to the state BEFORE action[i] was executed
        hist_obs_data = self.humanoid_obs_hist_buf.get_all(env_ids)[:actual_steps]
        hist_action_data = self.humanoid_action_hist_buf.get_all(env_ids)[:actual_steps]

        # Add time embeddings if enabled
        config = self.config.historical_self_obs_with_actions
        if config.with_time:
            data_steps = hist_obs_data.shape[0]
            # Use simple time offsets like historical_self_obs (negative times, starting from 0)
            time_offsets = -self.env.dt * torch.arange(data_steps, device=self.env.device, dtype=torch.float)
            time_embeddings = time_offsets.unsqueeze(1).expand(-1, hist_obs_data.shape[1]).unsqueeze(-1)

            # Concatenate obs, actions, and time for each step
            hist_obs_with_time = torch.cat([hist_obs_data, hist_action_data, time_embeddings], dim=-1)

            # Store in sequence format: [envs, steps, obs_per_step]
            self.historical_self_obs_with_actions[env_ids, :data_steps] = hist_obs_with_time.permute(1, 0, 2)
        else:
            # Concatenate obs and actions for each step
            hist_combined = torch.cat([hist_obs_data, hist_action_data], dim=-1)

            # Store in sequence format: [envs, steps, obs_per_step]
            self.historical_self_obs_with_actions[env_ids, :actual_steps] = hist_combined.permute(1, 0, 2)

    def build_self_obs_demo(
        self, motion_ids: Tensor, motion_times0: Tensor, num_steps: int
    ):
        dt = self.env.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.env.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)

        lengths = self.env.motion_lib.state.motion_lengths[motion_ids]

        motion_times = motion_times.view(-1).clamp(max=lengths).clamp(min=0)

        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)

        obs_demo = compute_humanoid_observations_max(
            ref_state.rigid_body_pos,
            ref_state.rigid_body_rot,
            ref_state.rigid_body_vel,
            ref_state.rigid_body_ang_vel,
            torch.zeros(len(motion_ids), 1, device=self.env.device),
            self.config.local_root_obs,
            self.config.root_height_obs,
            True,
        )
        return obs_demo

    def get_obs(self):
        obs_dict = {
            "self_obs": self.humanoid_obs.clone(),
            "historical_self_obs": self.humanoid_obs_hist_buf.get_all_flattened().clone(),
        }

        # Add historical_actions and historical_self_obs_with_actions only if enabled
        if self.config.historical_self_obs_with_actions.enabled:
            obs_dict["historical_actions"] = self.humanoid_action_hist_buf.get_all_flattened().clone()
            # Always return fixed length tensor, but only use meaningful data
            # The rest is zero-padded and will be handled by transformer
            obs_dict["historical_self_obs_with_actions"] = self.historical_self_obs_with_actions.reshape(self.env.num_envs, -1)

        return obs_dict
