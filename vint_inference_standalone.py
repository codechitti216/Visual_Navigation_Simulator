#!/usr/bin/env python3
"""
Standalone ViNT Inference Script for TORCS Data
Run ViNT navigation model on captured TORCS frames without ROS
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

class StandaloneViNTInference:
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
        latest = sorted(sessions)[-1]  # Get the latest by name (timestamp-based)
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
        
        print(f"Loaded {len(self.frame_paths)} frames and {len(self.pose_data)} pose entries")
        
    def load_frame_sequence(self, start_idx, context_size):
        """Load a sequence of frames for model input"""
        frames = []
        for i in range(max(0, start_idx - context_size + 1), start_idx + 1):
            if i < len(self.frame_paths):
                img = PILImage.open(self.frame_paths[i])
                frames.append(img)
            else:
                # Pad with first frame if we don't have enough history
                img = PILImage.open(self.frame_paths[0])
                frames.append(img)
        return frames
    
    def predict_waypoint(self, frame_idx):
        """Run ViNT inference on a single frame"""
        context_size = self.model_params.get("context_size", 5)
        
        # Load frame sequence
        frames = self.load_frame_sequence(frame_idx, context_size)
        
        # Transform images for model input
        obs_images = transform_images(frames, self.model_params["image_size"], center_crop=False)
        obs_images = obs_images.to(self.device)
        
        # For exploration mode (no specific goal), use a fake goal
        fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
        mask = torch.ones(1).long().to(self.device)  # ignore the goal
        
        with torch.no_grad():
            # Encode vision features
            obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, input_goal_mask=mask)
            
            # For diffusion models (like NoMaD), we need the diffusion process
            if self.model_params.get("model_type") == "nomad":
                return self.predict_with_diffusion(obs_cond)
            else:
                # For direct regression models (like ViNT)
                waypoints = self.model('action_decoder', obs_cond=obs_cond)
                waypoints = to_numpy(waypoints[0])  # Get first sample
                return waypoints
    
    def predict_with_diffusion(self, obs_cond, num_samples=1):
        """Run diffusion-based prediction (for NoMaD)"""
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        num_diffusion_iters = self.model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # Repeat observation conditioning for multiple samples
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(num_samples, 1, 1)
        
        # Initialize action from Gaussian noise
        noisy_action = torch.randn(
            (num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
        naction = noisy_action
        
        # Initialize scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)
        
        # Diffusion denoising loop
        for k in noise_scheduler.timesteps[:]:
            # Predict noise
            noise_pred = self.model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )
            # Remove noise
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        
        naction = to_numpy(get_action(naction))
        return naction[0]  # Return first sample
    
    def run_inference_on_trajectory(self, num_frames=50, save_results=True):
        """Run ViNT inference on a sequence of TORCS frames"""
        results = []
        
        print(f"Running ViNT inference on {num_frames} frames...")
        
        for i in range(min(num_frames, len(self.frame_paths))):
            print(f"Processing frame {i+1}/{num_frames}", end='\r')
            
            # Get ground truth data
            pose = self.pose_data.iloc[i]
            goal = self.goal_data.iloc[i]
            
            # Run ViNT prediction
            try:
                predicted_waypoints = self.predict_waypoint(i)
                
                # Store results
                result = {
                    'frame_id': i,
                    'timestamp': pose['timestamp'],
                    'gt_x': pose['x'],
                    'gt_y': pose['y'],
                    'gt_theta': pose['theta'],
                    'gt_speed': pose['speed'],
                    'goal_x': goal['goal_x'],
                    'goal_y': goal['goal_y'],
                    'pred_waypoints': predicted_waypoints.tolist() if isinstance(predicted_waypoints, np.ndarray) else predicted_waypoints
                }
                results.append(result)
                
            except Exception as e:
                print(f"\nError processing frame {i}: {e}")
                continue
        
        print(f"\nCompleted inference on {len(results)} frames")
        
        if save_results:
            # Save results
            results_file = os.path.join(self.data_dir, "vint_inference_results.csv")
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to: {results_file}")
        
        return results
    
    def visualize_results(self, results, save_plot=True):
        """Create visualization of ViNT predictions vs ground truth"""
        if not results:
            print("No results to visualize")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract trajectory
        gt_x = [r['gt_x'] for r in results]
        gt_y = [r['gt_y'] for r in results]
        
        # Plot ground truth trajectory
        ax1.plot(gt_x, gt_y, 'b-', label='Ground Truth Trajectory', linewidth=2)
        ax1.scatter(gt_x[0], gt_y[0], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(gt_x[-1], gt_y[-1], color='red', s=100, label='End', zorder=5)
        
        # Plot goal points
        goal_x = [r['goal_x'] for r in results]
        goal_y = [r['goal_y'] for r in results]
        ax1.scatter(goal_x, goal_y, color='orange', alpha=0.5, s=30, label='Goals')
        
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title('TORCS Trajectory with ViNT Goals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot speed over time
        speeds = [r['gt_speed'] for r in results]
        frame_ids = [r['frame_id'] for r in results]
        ax2.plot(frame_ids, speeds, 'g-', linewidth=2)
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Speed Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = os.path.join(self.data_dir, "vint_analysis.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {plot_file}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Standalone ViNT inference on TORCS data")
    parser.add_argument("--model", default="vint", help="Model name (vint, gnm, nomad)")
    parser.add_argument("--data-dir", default="vint_torcs_logs", help="TORCS data directory")
    parser.add_argument("--num-frames", type=int, default=50, help="Number of frames to process")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    
    args = parser.parse_args()
    
    # Run ViNT inference
    vint = StandaloneViNTInference(args.model, args.data_dir)
    results = vint.run_inference_on_trajectory(args.num_frames)
    
    if args.visualize:
        vint.visualize_results(results)
    
    print("ViNT inference completed!")
    print(f"Processed {len(results)} frames")
    print(f"Results saved in: {vint.data_dir}")

if __name__ == "__main__":
    main() 