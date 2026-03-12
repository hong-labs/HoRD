"""
Motion recording utility module.

Provides utilities to record and save motion data:
- BatchMotionRecorder: records motion data for multiple environments in batch.
"""

import torch
from pathlib import Path


class BatchMotionRecorder:
    """Record motion data for multiple environments in batch."""
    
    def __init__(self, simulator, num_envs=64, num_frames=120):
        self.simulator = simulator
        self.num_envs = num_envs
        self.num_frames = num_frames
        self.dt = simulator.dt
        self.recorded_data = {
            "global_translation": [],
            "global_rotation": [],
            "dof_pos": [],
            "dof_vel": [],
            "rigid_body_vel": [],
            "rigid_body_ang_vel": []
        }
        self.frame_count = 0
        
    def record_frame(self):
        """Record all environment states for the current frame."""
        if self.frame_count >= self.num_frames:
            return False
            
        # Get rigid-body states
        bodies_state = self.simulator.get_bodies_state()
        self.recorded_data["global_translation"].append(bodies_state.rigid_body_pos)
        self.recorded_data["global_rotation"].append(bodies_state.rigid_body_rot)
        self.recorded_data["rigid_body_vel"].append(bodies_state.rigid_body_vel)
        self.recorded_data["rigid_body_ang_vel"].append(bodies_state.rigid_body_ang_vel)
        
        # Get DOF states
        dof_state = self.simulator.get_dof_state()
        self.recorded_data["dof_pos"].append(dof_state.dof_pos)
        self.recorded_data["dof_vel"].append(dof_state.dof_vel)
        
        self.frame_count += 1
        return True
        
    def save_motions(self, output_path):
        env_id = 0
        # Extract data
        dof_pos = torch.stack(self.recorded_data["dof_pos"])[:, env_id].detach().cpu()
        dof_vel = torch.stack(self.recorded_data["dof_vel"])[:, env_id].detach().cpu()
        global_translation = torch.stack(self.recorded_data["global_translation"])[:, env_id].detach().cpu()
        global_rotation = torch.stack(self.recorded_data["global_rotation"])[:, env_id].detach().cpu()
        global_velocity = torch.stack(self.recorded_data["rigid_body_vel"])[:, env_id].detach().cpu()
        global_angular_velocity = torch.stack(self.recorded_data["rigid_body_ang_vel"])[:, env_id].detach().cpu()

        # Extract root linear and angular velocity
        global_root_velocity = global_velocity[:, 0, :]
        global_root_angular_velocity = global_angular_velocity[:, 0, :]

        # Create time series
        motion_time = torch.arange(self.num_frames, dtype=torch.float32) * self.dt

        motion = {
            "dof_pos": dof_pos,
            "dof_vels": dof_vel,
            "global_translation": global_translation,
            "global_rotation": global_rotation,
            "global_velocity": global_velocity,
            "global_angular_velocity": global_angular_velocity,
            "global_root_velocity": global_root_velocity,
            "global_root_angular_velocity": global_root_angular_velocity,
            "motion_time": motion_time,
            "fps": 1.0 / self.dt,
            "num_frames": self.num_frames,
        }

        torch.save(motion, output_path)
        print(f"Motion data saved: {self.num_frames} frames, {self.num_frames * self.dt:.1f}s, {1.0/self.dt:.0f}fps")
        
        return True
