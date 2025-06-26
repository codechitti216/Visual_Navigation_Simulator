#!/usr/bin/env python3
"""
TORCS Trajectory Inference with ViNT/NoMaD
- Goal: Last image in trajectory
- Start: 6th image (with 5 past observations)
- Output: Predicted waypoints overlaid on current observation
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add ViNT modules to path
sys.path.append('visualnav-transformer/deployment/src')
sys.path.append('visualnav-transformer/train')

from utils import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action

class TORCSTrajectoryInference:
    def __init__(self, model_name="vint", data_dir="vint_torcs_logs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model configuration
        models_config_path = "visualnav-transformer/deployment/config/models.yaml"
        with open(models_config_path, "r") as f:
            models_config = yaml.safe_load(f)
        
        # Get model paths
        model_config_path = models_config[model_name]["config_path"]
        model_config_path = f"visualnav-transformer/deployment/{model_config_path}"
        
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)
        
        # Load model weights
        ckpt_path = models_config[model_name]["ckpt_path"] 
        ckpt_path = f"visualnav-transformer/deployment/{ckpt_path}"
        
        print(f"Loading model from {ckpt_path}")
        self.model = load_model(ckpt_path, self.model_params, self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Find the latest session directory
        self.data_dir = self.find_latest_session(data_dir)
        print(f"Using data from: {self.data_dir}")
        
        # Load TORCS data
        self.load_torcs_data()
        
    def find_latest_session(self, base_dir):
        """Find the most recent TORCS session directory"""
        sessions = [d for d in os.listdir(base_dir) if d.startswith("Ushite-city")]
        if not sessions:
            raise ValueError(f"No TORCS sessions found in {base_dir}")
        latest = sorted(sessions)[-1]
        return os.path.join(base_dir, latest)
    
    def load_torcs_data(self):
        """Load TORCS trajectory and image data"""
        # Load pose data
        pose_file = os.path.join(self.data_dir, "pose.csv")
        self.pose_data = pd.read_csv(pose_file)
        
        # Load goal data  
        goal_file = os.path.join(self.data_dir, "goal.csv")
        self.goal_data = pd.read_csv(goal_file)
        
        # Get frame paths
        frames_dir = os.path.join(self.data_dir, "frames")
        self.frame_paths = sorted([
            os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
            if f.endswith('.png')
        ])
        
        # Goal image is the LAST frame
        self.goal_image = PILImage.open(self.frame_paths[-1])
        
        print(f"Loaded {len(self.frame_paths)} frames")
        print(f"Goal image: {self.frame_paths[-1]}")
        
    def load_frame_sequence(self, current_idx, context_size=5):
        """Load context frames for current observation"""
        frames = []
        start_idx = max(0, current_idx - context_size)
        
        for i in range(start_idx, current_idx + 1):
            if i < len(self.frame_paths):
                img = PILImage.open(self.frame_paths[i])
                frames.append(img)
        
        # Ensure we have exactly context_size + 1 frames
        while len(frames) < context_size + 1:
            frames.insert(0, frames[0])  # Pad with first frame
            
        return frames
    
    def predict_waypoints(self, current_frame_idx):
        """Run ViNT inference with goal conditioning"""
        context_size = self.model_params.get("context_size", 5)
        
        # Load observation sequence (past + current)
        obs_frames = self.load_frame_sequence(current_frame_idx, context_size)
        
        # Transform images
        obs_images = transform_images(obs_frames, self.model_params["image_size"], center_crop=False)
        goal_image = transform_images([self.goal_image], self.model_params["image_size"], center_crop=False)
        
        obs_images = obs_images.to(self.device)
        goal_image = goal_image.to(self.device)
        
        # No goal masking - we want goal-conditioned behavior
        mask = torch.zeros(1).long().to(self.device)
        
        with torch.no_grad():
            # Get vision encoding
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=goal_image, input_goal_mask=mask)
            
            if self.model_params.get("model_type") == "nomad":
                # Use diffusion for NoMaD
                waypoints = self.predict_with_diffusion(obs_cond)
            else:
                # Direct prediction for ViNT
                waypoints = self.model('dist_pred_net', obsgoal_cond=obs_cond.flatten(start_dim=1))
                waypoints = to_numpy(waypoints[0])
                
        return waypoints
    
    def predict_with_diffusion(self, obs_cond, num_samples=1):
        """NoMaD diffusion inference"""
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        num_diffusion_iters = self.model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(num_samples, 1, 1)
        
        # Initialize action from noise
        noisy_action = torch.randn(
            (num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
        naction = noisy_action
        
        noise_scheduler.set_timesteps(num_diffusion_iters)
        
        for k in noise_scheduler.timesteps[:]:
            noise_pred = self.model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        # Convert to actual actions
        actions = get_action(naction, self.model_params)
        return to_numpy(actions[0])
    
    def visualize_prediction(self, frame_idx, waypoints, save_path=None):
        """Plot predicted waypoints on current observation"""
        # Load current frame
        current_img = PILImage.open(self.frame_paths[frame_idx])
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Current observation with waypoints
        axes[0].imshow(current_img)
        
        # Convert waypoints to image coordinates (this is approximate)
        # You may need to adjust based on your camera model
        if len(waypoints.shape) == 2:  # Multiple waypoints
            for i, (dx, dy) in enumerate(waypoints):
                # Simple projection: scale deltas to image coordinates
                x = current_img.width / 2 + dx * 50  # Scale factor
                y = current_img.height - dy * 50
                
                # Color code waypoints (near to far: red to blue)
                color = plt.cm.coolwarm(i / len(waypoints))
                axes[0].plot(x, y, 'o', color=color, markersize=8)
                axes[0].text(x + 5, y, f't+{i+1}', color=color, fontweight='bold')
        
        axes[0].set_title(f'Frame {frame_idx}: Predicted Waypoints')
        axes[0].axis('off')
        
        # Plot 2: Past 5 frames
        past_frames = self.load_frame_sequence(frame_idx, 5)
        if len(past_frames) >= 5:
            for i in range(5):
                ax_past = plt.subplot2grid((1, 15), (0, 3 + i * 2), colspan=2)
                ax_past.imshow(past_frames[i])
                ax_past.set_title(f't-{5-i}')
                ax_past.axis('off')
        
        # Plot 3: Goal image
        axes[2].imshow(self.goal_image)
        axes[2].set_title('Goal Image')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
    def run_trajectory_inference(self, start_frame=5, num_frames=10):
        """Run inference starting from 6th frame (index 5)"""
        print(f"Running trajectory inference:")
        print(f"- Start frame: {start_frame} (6th frame)")
        print(f"- Number of frames: {num_frames}")
        print(f"- Goal: Last frame ({len(self.frame_paths)-1})")
        
        results = []
        
        for i in range(start_frame, min(start_frame + num_frames, len(self.frame_paths) - 1)):
            print(f"Processing frame {i}...")
            
            # Run inference
            waypoints = self.predict_waypoints(i)
            
            # Get ground truth data
            pose = self.pose_data.iloc[i] if i < len(self.pose_data) else None
            
            # Create visualization
            save_path = f"frame_{i}_prediction.png"
            self.visualize_prediction(i, waypoints, save_path)
            
            result = {
                'frame_idx': i,
                'predicted_waypoints': waypoints.tolist() if hasattr(waypoints, 'tolist') else waypoints,
                'gt_pose': pose.to_dict() if pose is not None else None
            }
            results.append(result)
        
        print(f"Completed inference on {len(results)} frames")
        return results

def main():
    parser = argparse.ArgumentParser(description="TORCS Trajectory Inference")
    parser.add_argument("--model", default="vint", choices=["vint", "nomad"], 
                       help="Model to use for inference")
    parser.add_argument("--start-frame", type=int, default=5,
                       help="Starting frame index (default: 5 for 6th frame)")
    parser.add_argument("--num-frames", type=int, default=10,
                       help="Number of frames to process")
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = TORCSTrajectoryInference(model_name=args.model)
    
    # Run inference
    results = inferencer.run_trajectory_inference(
        start_frame=args.start_frame, 
        num_frames=args.num_frames
    )
    
    print("Trajectory inference completed!")

if __name__ == "__main__":
    main() 