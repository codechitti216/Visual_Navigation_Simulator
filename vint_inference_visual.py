#!/usr/bin/env python3
"""
Enhanced ViNT Inference with Rich Visualizations for TORCS Data
- Current frame with next 5 waypoints overlaid
- Past 5 frames sequence
- Goal visualization
- Temporal distance annotations
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import cv2
import argparse
from pathlib import Path

# Add ViNT modules to path
sys.path.append('visualnav-transformer/deployment/src')
sys.path.append('visualnav-transformer/train')

from utils import load_model, transform_images, to_numpy
from vint_train.training.train_utils import get_action

class EnhancedViNTVisualizer:
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
        
        # Setup visualization colors for waypoints
        self.waypoint_colors = [
            (255, 0, 0),    # Red - immediate next
            (255, 165, 0),  # Orange
            (255, 255, 0),  # Yellow  
            (0, 255, 0),    # Green
            (0, 0, 255)     # Blue - furthest
        ]
        
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
        
        print(f"Loaded {len(self.frame_paths)} frames and {len(self.pose_data)} pose entries")
        
    def load_frame_sequence(self, start_idx, context_size):
        """Load a sequence of frames for model input"""
        frames = []
        for i in range(max(0, start_idx - context_size + 1), start_idx + 1):
            if i < len(self.frame_paths):
                img = PILImage.open(self.frame_paths[i])
                frames.append(img)
            else:
                img = PILImage.open(self.frame_paths[0])
                frames.append(img)
        return frames
    
    def predict_waypoint(self, frame_idx):
        """Run ViNT inference on a single frame and return waypoints with temporal predictions"""
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
                waypoints, diffusion_images = self.predict_with_diffusion(obs_cond, return_intermediates=True)
                return waypoints, diffusion_images
            else:
                # For direct regression models (like ViNT)
                waypoints = self.model('action_decoder', obs_cond=obs_cond)
                waypoints = to_numpy(waypoints[0])  # Get first sample
                
                # ViNT typically predicts temporal distances based on the trajectory length
                # Extract temporal predictions if available in model output
                temporal_distances = self.extract_temporal_distances(waypoints)
                return waypoints, temporal_distances, None
    
    def predict_with_diffusion(self, obs_cond, num_samples=1, return_intermediates=False):
        """Run diffusion-based prediction (for NoMaD) and optionally return intermediate images"""
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
        
        # Store intermediate states for visualization
        intermediate_states = []
        
        # Diffusion denoising loop
        for i, k in enumerate(noise_scheduler.timesteps[:]):
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
            
            # Store intermediate states for visualization (every 5 steps)
            if return_intermediates and i % 5 == 0:
                intermediate_states.append(to_numpy(naction[0].clone()))
        
        final_action = to_numpy(get_action(naction))
        
        if return_intermediates:
            return final_action[0], intermediate_states
        else:
            return final_action[0]
    
    def extract_temporal_distances(self, waypoints):
        """Extract temporal distances from waypoint predictions"""
        # For ViNT, temporal distances are typically based on the prediction horizon
        # This is a simplified version - you may need to adjust based on your model
        if hasattr(self.model_params, 'prediction_horizon'):
            total_time = self.model_params['prediction_horizon']
        else:
            # Default assumption: 2 seconds total prediction horizon
            total_time = 2.0
        
        num_waypoints = len(waypoints)
        if num_waypoints > 0:
            temporal_distances = [i * (total_time / num_waypoints) for i in range(num_waypoints)]
        else:
            temporal_distances = []
        
        return temporal_distances
    
    def world_to_image_coordinates(self, world_x, world_y, current_pose, image_size=(512, 512)):
        """
        Convert world coordinates to image pixel coordinates
        This is a simplified projection - you might need to adjust based on TORCS camera parameters
        """
        # Get current car position and orientation
        car_x, car_y, car_theta = current_pose['x'], current_pose['y'], current_pose['theta']
        
        # Transform to car-relative coordinates
        relative_x = world_x - car_x
        relative_y = world_y - car_y
        
        # Rotate to car's frame of reference
        cos_theta = np.cos(-car_theta)
        sin_theta = np.sin(-car_theta)
        
        local_x = relative_x * cos_theta - relative_y * sin_theta
        local_y = relative_x * sin_theta + relative_y * cos_theta
        
        # Simple perspective projection (you may need to adjust these parameters)
        # Assuming camera is looking forward, with some field of view
        scale = 20  # Adjust this based on your needs
        center_x, center_y = image_size[0] // 2, image_size[1] // 2
        
        # Project to image coordinates
        pixel_x = center_x + int(local_y * scale)  # Note: y becomes x in image (side-to-side)
        pixel_y = center_y - int(local_x * scale)  # Note: x becomes -y in image (forward becomes up)
        
        return pixel_x, pixel_y
    
    def overlay_waypoints_on_frame(self, frame_img, waypoints, current_pose, timestamps):
        """Overlay predicted waypoints on the current frame"""
        # Convert PIL to OpenCV format
        frame_cv = cv2.cvtColor(np.array(frame_img), cv2.COLOR_RGB2BGR)
        
        # Draw waypoints
        for i, waypoint in enumerate(waypoints[:5]):  # Only show first 5 waypoints
            if len(waypoint) >= 2:
                # For now, we'll place waypoints as relative positions
                # This is a simplified visualization - you might want to improve this
                center_x, center_y = frame_cv.shape[1] // 2, frame_cv.shape[0] // 2
                
                # Scale waypoints to image coordinates (simplified)
                scale = 100
                pixel_x = center_x + int(waypoint[0] * scale)
                pixel_y = center_y - int(waypoint[1] * scale)
                
                # Ensure coordinates are within image bounds
                pixel_x = max(0, min(pixel_x, frame_cv.shape[1] - 1))
                pixel_y = max(0, min(pixel_y, frame_cv.shape[0] - 1))
                
                # Draw waypoint circle
                color = self.waypoint_colors[i] if i < len(self.waypoint_colors) else (255, 255, 255)
                cv2.circle(frame_cv, (pixel_x, pixel_y), 8, color, -1)
                cv2.circle(frame_cv, (pixel_x, pixel_y), 10, (0, 0, 0), 2)  # Black border
                
                # Add temporal distance annotation
                if i < len(timestamps):
                    time_diff = timestamps[i] - timestamps[0] if timestamps[0] > 0 else i * 0.1
                    text = f"t+{time_diff:.1f}s"
                    cv2.putText(frame_cv, text, (pixel_x + 15, pixel_y - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB
        return cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    
    def create_goal_visualization(self, current_pose, goal_pos, trajectory_data):
        """Create a top-down view showing goal and trajectory"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot trajectory
        trajectory_x = trajectory_data['x']
        trajectory_y = trajectory_data['y']
        ax.plot(trajectory_x, trajectory_y, 'b-', alpha=0.5, linewidth=2, label='Trajectory')
        
        # Mark current position
        ax.scatter(current_pose['x'], current_pose['y'], color='red', s=100, 
                  marker='o', label='Current Position', zorder=5)
        
        # Mark goal position
        ax.scatter(goal_pos['goal_x'], goal_pos['goal_y'], color='green', s=100, 
                  marker='*', label='Goal Position', zorder=5)
        
        # Draw orientation arrow for current position
        arrow_length = 5
        dx = arrow_length * np.cos(current_pose['theta'])
        dy = arrow_length * np.sin(current_pose['theta'])
        ax.arrow(current_pose['x'], current_pose['y'], dx, dy, 
                head_width=2, head_length=1, fc='red', ec='red')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Goal Visualization - Top View')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return fig
    
    def create_comprehensive_visualization(self, frame_idx, save_output=True):
        """Create the complete visualization with all requested components"""
        # Get current frame and data
        current_frame = PILImage.open(self.frame_paths[frame_idx])
        current_pose = self.pose_data.iloc[frame_idx]
        current_goal = self.goal_data.iloc[frame_idx]
        
        # Get past 5 frames
        past_frames = []
        for i in range(max(0, frame_idx - 4), frame_idx):
            if i < len(self.frame_paths):
                past_frames.append(PILImage.open(self.frame_paths[i]))
        
        # Run ViNT inference
        predicted_waypoints = self.predict_waypoint(frame_idx)
        
        # Create timestamps for waypoints (simplified)
        base_timestamp = current_pose['timestamp']
        waypoint_timestamps = [base_timestamp + i * 0.5 for i in range(len(predicted_waypoints))]
        
        # Overlay waypoints on current frame
        annotated_frame = self.overlay_waypoints_on_frame(
            current_frame, predicted_waypoints, current_pose, waypoint_timestamps)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. Current frame with waypoints (large, top-left)
        ax_current = fig.add_subplot(gs[0:2, 0:3])
        ax_current.imshow(annotated_frame)
        ax_current.set_title(f'Current Frame {frame_idx} with Next 5 Waypoints\n'
                           f'Speed: {current_pose["speed"]:.1f} m/s', fontsize=14)
        ax_current.axis('off')
        
        # Add waypoint legend
        legend_text = "Waypoints:\n"
        for i, color in enumerate(self.waypoint_colors):
            legend_text += f"● t+{i*0.5:.1f}s\n"
        ax_current.text(0.02, 0.98, legend_text, transform=ax_current.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Past 5 frames (top-right)
        for i, past_frame in enumerate(past_frames[-5:]):  # Show last 5
            ax_past = fig.add_subplot(gs[0, 3+i] if i < 3 else gs[1, i-3+3])
            ax_past.imshow(past_frame)
            ax_past.set_title(f'Frame {frame_idx-len(past_frames)+i+1}', fontsize=10)
            ax_past.axis('off')
        
        # 3. Goal visualization (bottom-left)
        ax_goal = fig.add_subplot(gs[2, 0:3])
        
        # Create trajectory plot
        start_idx = max(0, frame_idx - 50)
        end_idx = min(len(self.pose_data), frame_idx + 20)
        traj_data = self.pose_data.iloc[start_idx:end_idx]
        
        ax_goal.plot(traj_data['x'], traj_data['y'], 'b-', alpha=0.5, linewidth=2, label='Trajectory')
        ax_goal.scatter(current_pose['x'], current_pose['y'], color='red', s=100, 
                       marker='o', label='Current Position', zorder=5)
        ax_goal.scatter(current_goal['goal_x'], current_goal['goal_y'], color='green', s=100, 
                       marker='*', label='Goal Position', zorder=5)
        
        # Draw orientation arrow
        arrow_length = 2
        dx = arrow_length * np.cos(current_pose['theta'])
        dy = arrow_length * np.sin(current_pose['theta'])
        ax_goal.arrow(current_pose['x'], current_pose['y'], dx, dy, 
                     head_width=1, head_length=0.5, fc='red', ec='red')
        
        ax_goal.set_xlabel('X Position')
        ax_goal.set_ylabel('Y Position')
        ax_goal.set_title('Goal Visualization - Top View')
        ax_goal.legend()
        ax_goal.grid(True, alpha=0.3)
        ax_goal.axis('equal')
        
        # 4. Predicted waypoints data (bottom-right)
        ax_data = fig.add_subplot(gs[2, 3:])
        ax_data.axis('off')
        
        # Create waypoints data table
        waypoint_text = "Predicted Waypoints:\n\n"
        for i, wp in enumerate(predicted_waypoints[:5]):
            if len(wp) >= 2:
                waypoint_text += f"Waypoint {i+1}: ({wp[0]:.3f}, {wp[1]:.3f})\n"
                waypoint_text += f"Time: t+{i*0.5:.1f}s\n\n"
        
        # Add current state info
        state_text = f"\nCurrent State:\n"
        state_text += f"Position: ({current_pose['x']:.1f}, {current_pose['y']:.1f})\n"
        state_text += f"Orientation: {current_pose['theta']:.3f} rad\n"
        state_text += f"Speed: {current_pose['speed']:.2f} m/s\n"
        state_text += f"Goal: ({current_goal['goal_x']:.1f}, {current_goal['goal_y']:.1f})\n"
        
        ax_data.text(0.05, 0.95, waypoint_text + state_text, transform=ax_data.transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.suptitle(f'ViNT Inference Visualization - Frame {frame_idx}', fontsize=16)
        
        if save_output:
            output_file = os.path.join(self.data_dir, f"vint_visualization_frame_{frame_idx:04d}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Visualization saved: {output_file}")
        
        return fig

def main():
    parser = argparse.ArgumentParser(description="Enhanced ViNT inference with rich visualizations")
    parser.add_argument("--model", default="vint", help="Model name (vint, gnm, nomad)")
    parser.add_argument("--data-dir", default="vint_torcs_logs", help="TORCS data directory")
    parser.add_argument("--frame", type=int, default=50, help="Frame to visualize")
    parser.add_argument("--sequence", action="store_true", help="Generate sequence of visualizations")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames for sequence")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = EnhancedViNTVisualizer(args.model, args.data_dir)
    
    if args.sequence:
        print(f"Generating visualization sequence for {args.num_frames} frames...")
        for i in range(args.frame, min(args.frame + args.num_frames, len(visualizer.frame_paths))):
            print(f"Processing frame {i}...")
            fig = visualizer.create_comprehensive_visualization(i)
            plt.close(fig)  # Close to save memory
    else:
        # Single frame visualization
        print(f"Creating visualization for frame {args.frame}...")
        fig = visualizer.create_comprehensive_visualization(args.frame)
        plt.show()
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 