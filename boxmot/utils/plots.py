# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from typing import Dict, List, Tuple, Optional, Union
import cv2
from collections import defaultdict, deque

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class EnhancedMetricsPlotter:
    """
    Enhanced plotting class for comprehensive tracking metrics visualization including occlusion analysis.
    
    Features:
    - Traditional radar and line charts
    - Occlusion analysis plots
    - Track trajectory visualization
    - Real-time performance monitoring
    - Quality assessment plots
    - Comparison visualizations
    """

    def __init__(self, root_folder: str = '.', style: str = 'seaborn'):
        """
        Initialize the Enhanced Metrics Plotter.

        Parameters
        ----------
        root_folder : str
            Directory to save plots
        style : str
            Matplotlib style ('seaborn', 'dark', 'classic')
        """
        self.root_folder = root_folder
        os.makedirs(self.root_folder, exist_ok=True)
        
        # Set plotting style
        if style == 'seaborn' and SEABORN_AVAILABLE:
            plt.style.use('seaborn-v0_8')
        elif style == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        # Color palettes
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.occlusion_colors = {
            'visible': '#2E8B57',
            'partial': '#FFA500', 
            'occluded': '#DC143C',
            'crowd': '#8B0000'
        }

    def plot_radar_chart(self,
                         data: dict,
                         labels: list,
                         title: str = 'Enhanced Tracking Performance',
                         figsize: tuple = (8, 8),
                         ylim: tuple = (0, 100.0),
                         yticks: list = None,
                         ytick_labels: list = None,
                         filename: str = 'enhanced_radar_chart.png'):
        """
        Plot enhanced radar chart with improved styling and occlusion metrics.
        
        Parameters
        ----------
        data : dict
            Dictionary with method names as keys and metric lists as values
        labels : list
            List of metric names
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Compute angle for each metric
        angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
        angles += angles[:1]  # Complete the circle
        
        # Plot each method
        for i, (method, values) in enumerate(data.items()):
            values += values[:1]  # Complete the circle
            color = self.colors[i % len(self.colors)]
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method, color=color, markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(ylim)
        
        if yticks:
            ax.set_yticks(yticks)
            if ytick_labels:
                ax.set_yticklabels(ytick_labels)
        
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_occlusion_analysis(self,
                               occlusion_data: Dict,
                               filename: str = 'occlusion_analysis.png',
                               figsize: tuple = (15, 10)):
        """
        Plot comprehensive occlusion analysis including timeline and distribution.
        
        Parameters
        ----------
        occlusion_data : dict
            Dictionary containing occlusion statistics and timeline data
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Occlusion Timeline
        frames = occlusion_data.get('frames', [])
        occlusion_levels = occlusion_data.get('occlusion_levels', [])
        track_counts = occlusion_data.get('track_counts', [])
        
        ax1.plot(frames, occlusion_levels, color=self.occlusion_colors['partial'], 
                linewidth=2, label='Occlusion Level')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(frames, track_counts, color='blue', 
                     linewidth=2, linestyle='--', label='Track Count')
        
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Average Occlusion Level', color=self.occlusion_colors['partial'])
        ax1_twin.set_ylabel('Track Count', color='blue')
        ax1.set_title('Occlusion Timeline Analysis')
        ax1.grid(True, alpha=0.3)
        
        # 2. Occlusion Type Distribution
        occlusion_types = occlusion_data.get('occlusion_types', {})
        if occlusion_types:
            labels = list(occlusion_types.keys())
            sizes = list(occlusion_types.values())
            colors = [self.occlusion_colors.get(label.lower(), '#888888') for label in labels]
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Occlusion Type Distribution')
        
        # 3. ID Switch Analysis
        id_switches = occlusion_data.get('id_switches_per_frame', [])
        if id_switches:
            ax3.bar(range(len(id_switches)), id_switches, 
                   color=self.occlusion_colors['occluded'], alpha=0.7)
            ax3.set_xlabel('Frame Groups (10-frame bins)')
            ax3.set_ylabel('ID Switches')
            ax3.set_title('ID Switches Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Track Quality vs Occlusion
        track_qualities = occlusion_data.get('track_qualities', [])
        track_occlusions = occlusion_data.get('track_occlusions', [])
        
        if track_qualities and track_occlusions:
            scatter = ax4.scatter(track_occlusions, track_qualities, 
                                c=track_occlusions, cmap='RdYlGn_r', 
                                alpha=0.6, s=50)
            ax4.set_xlabel('Occlusion Level')
            ax4.set_ylabel('Track Quality')
            ax4.set_title('Track Quality vs Occlusion Level')
            plt.colorbar(scatter, ax=ax4, label='Occlusion Level')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_track_trajectories(self,
                               track_data: Dict,
                               image_size: Tuple[int, int],
                               filename: str = 'track_trajectories.png',
                               max_tracks: int = 20,
                               show_occlusion: bool = True):
        """
        Plot track trajectories with occlusion information.
        
        Parameters
        ----------
        track_data : dict
            Dictionary with track_id as keys and trajectory data as values
        image_size : tuple
            (width, height) of the tracking scene
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up the plot area to match image dimensions
        ax.set_xlim(0, image_size[0])
        ax.set_ylim(image_size[1], 0)  # Invert y-axis to match image coordinates
        ax.set_aspect('equal')
        
        # Plot trajectories for each track
        track_count = 0
        for track_id, trajectory in track_data.items():
            if track_count >= max_tracks:
                break
                
            positions = trajectory.get('positions', [])
            occlusion_levels = trajectory.get('occlusion_levels', [])
            frames = trajectory.get('frames', [])
            
            if not positions:
                continue
            
            # Extract x, y coordinates
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            
            # Color by occlusion level if available
            if show_occlusion and occlusion_levels:
                for i in range(len(positions) - 1):
                    occlusion = occlusion_levels[i]
                    if occlusion < 0.2:
                        color = self.occlusion_colors['visible']
                    elif occlusion < 0.6:
                        color = self.occlusion_colors['partial']
                    else:
                        color = self.occlusion_colors['occluded']
                    
                    ax.plot([x_coords[i], x_coords[i+1]], 
                           [y_coords[i], y_coords[i+1]], 
                           color=color, linewidth=2, alpha=0.8)
            else:
                color = self.colors[track_count % len(self.colors)]
                ax.plot(x_coords, y_coords, color=color, linewidth=2, 
                       label=f'Track {track_id}', alpha=0.8)
            
            # Mark start and end points
            if positions:
                ax.plot(x_coords[0], y_coords[0], 'o', color='green', 
                       markersize=8, label='Start' if track_count == 0 else "")
                ax.plot(x_coords[-1], y_coords[-1], 's', color='red', 
                       markersize=8, label='End' if track_count == 0 else "")
            
            track_count += 1
        
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_title('Track Trajectories with Occlusion Analysis')
        ax.grid(True, alpha=0.3)
        
        # Create legend for occlusion colors
        if show_occlusion:
            legend_elements = [
                plt.Line2D([0], [0], color=self.occlusion_colors['visible'], 
                          lw=3, label='Visible'),
                plt.Line2D([0], [0], color=self.occlusion_colors['partial'], 
                          lw=3, label='Partially Occluded'),
                plt.Line2D([0], [0], color=self.occlusion_colors['occluded'], 
                          lw=3, label='Fully Occluded'),
                plt.Line2D([0], [0], marker='o', color='green', lw=0, 
                          markersize=8, label='Track Start'),
                plt.Line2D([0], [0], marker='s', color='red', lw=0, 
                          markersize=8, label='Track End')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self,
                                  comparison_data: Dict,
                                  metrics: List[str],
                                  filename: str = 'performance_comparison.png',
                                  figsize: tuple = (14, 8)):
        """
        Plot side-by-side comparison of different tracker configurations.
        
        Parameters
        ----------
        comparison_data : dict
            Dictionary with configuration names and their metric values
        metrics : list
            List of metric names to compare
        """
        n_metrics = len(metrics)
        n_configs = len(comparison_data)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=False)
        if n_metrics == 1:
            axes = [axes]
        
        x_pos = np.arange(n_configs)
        bar_width = 0.8
        
        for i, metric in enumerate(metrics):
            values = [comparison_data[config].get(metric, 0) for config in comparison_data.keys()]
            
            bars = axes[i].bar(x_pos, values, bar_width, 
                              color=self.colors[i], alpha=0.7, 
                              edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            axes[i].set_xlabel('Configuration')
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(comparison_data.keys(), rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Tracker Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_real_time_metrics(self,
                              metrics_history: Dict,
                              window_size: int = 100,
                              filename: str = 'realtime_metrics.png',
                              figsize: tuple = (15, 10)):
        """
        Plot real-time tracking metrics with sliding window analysis.
        
        Parameters
        ----------
        metrics_history : dict
            Dictionary with metric names and their time series data
        window_size : int
            Size of sliding window for smoothing
        """
        n_metrics = len(metrics_history)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            frames = list(range(len(values)))
            
            # Plot raw values
            axes[i].plot(frames, values, alpha=0.3, color='gray', label='Raw')
            
            # Plot smoothed values using rolling average
            if len(values) > window_size:
                smoothed = []
                for j in range(len(values)):
                    start_idx = max(0, j - window_size // 2)
                    end_idx = min(len(values), j + window_size // 2)
                    smoothed.append(np.mean(values[start_idx:end_idx]))
                
                axes[i].plot(frames, smoothed, linewidth=2, 
                           color=self.colors[i], label='Smoothed')
            else:
                axes[i].plot(frames, values, linewidth=2, 
                           color=self.colors[i], label='Values')
            
            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f'{metric_name} Over Time')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Real-time Tracking Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_quality_heatmap(self,
                            quality_matrix: np.ndarray,
                            track_ids: List[int],
                            frame_range: Tuple[int, int],
                            filename: str = 'quality_heatmap.png',
                            figsize: tuple = (12, 8)):
        """
        Plot heatmap of track quality over time.
        
        Parameters
        ----------
        quality_matrix : np.ndarray
            Matrix of quality scores (tracks x frames)
        track_ids : list
            List of track IDs
        frame_range : tuple
            (start_frame, end_frame) for x-axis labeling
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(quality_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=0, vmax=1, interpolation='nearest')
        
        # Set labels and ticks
        ax.set_xlabel('Frame')
        ax.set_ylabel('Track ID')
        ax.set_title('Track Quality Heatmap Over Time')
        
        # Set y-axis ticks to track IDs
        ax.set_yticks(range(len(track_ids)))
        ax.set_yticklabels(track_ids)
        
        # Set x-axis ticks for frame range
        n_frames = quality_matrix.shape[1]
        frame_ticks = np.linspace(0, n_frames-1, min(10, n_frames), dtype=int)
        frame_labels = np.linspace(frame_range[0], frame_range[1], len(frame_ticks), dtype=int)
        ax.set_xticks(frame_ticks)
        ax.set_xticklabels(frame_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Quality Score', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.root_folder, filename), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def visualize_tracking_frame(img: np.ndarray,
                           tracks: List,
                           detections: List = None,
                           occlusion_info: Dict = None,
                           save_path: str = None,
                           show_trajectories: bool = True,
                           trajectory_length: int = 30) -> np.ndarray:
    """
    Enhanced frame visualization with occlusion information and track trajectories.
    
    Parameters
    ----------
    img : np.ndarray
        Input image
    tracks : list
        List of track objects
    detections : list, optional
        List of detection objects
    occlusion_info : dict, optional
        Dictionary with occlusion information per track
    save_path : str, optional
        Path to save the visualization
    show_trajectories : bool
        Whether to show track trajectories
    trajectory_length : int
        Number of previous positions to show in trajectory
    """
    vis_img = img.copy()
    
    # Color palettes
    track_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    occlusion_colors = {
        'visible': (46, 139, 87),      # SeaGreen
        'partial': (255, 165, 0),      # Orange  
        'occluded': (220, 20, 60),     # Crimson
        'crowd': (139, 0, 0)           # DarkRed
    }
    
    # Draw detections first (if provided)
    if detections:
        for det in detections:
            if hasattr(det, 'xyxy'):
                x1, y1, x2, y2 = det.xyxy.astype(int)
            else:
                x1, y1, w, h = det[:4].astype(int)
                x2, y2 = x1 + w, y1 + h
            
            # Draw detection box in light gray
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (128, 128, 128), 1)
    
    # Draw tracks with enhanced information
    for track in tracks:
        if not hasattr(track, 'is_confirmed') or not track.is_confirmed():
            continue
        
        # Get track bounding box
        if hasattr(track, 'to_tlbr'):
            x1, y1, x2, y2 = track.to_tlbr().astype(int)
        else:
            x1, y1, x2, y2 = track.xyxy.astype(int)
        
        track_id = getattr(track, 'id', 0)
        color_idx = track_id % len(track_colors)
        base_color = (np.array(track_colors[color_idx][:3]) * 255).astype(int)
        base_color = tuple(base_color.tolist())
        
        # Get occlusion information
        occlusion_level = 0.0
        occlusion_status = 'visible'
        if occlusion_info and track_id in occlusion_info:
            occlusion_level = occlusion_info[track_id].get('level', 0.0)
            if occlusion_level > 0.6:
                occlusion_status = 'occluded'
            elif occlusion_level > 0.2:
                occlusion_status = 'partial'
        
        # Draw occlusion overlay if applicable
        if occlusion_level > 0.1:
            overlay_color = occlusion_colors[occlusion_status]
            overlay_thickness = 3 if occlusion_level > 0.5 else 2
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), overlay_color, overlay_thickness)
        
        # Draw main track bounding box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), base_color, 2)
        
        # Draw trajectory if requested
        if show_trajectories and hasattr(track, 'trajectory'):
            trajectory = track.trajectory[-trajectory_length:]
            if len(trajectory) > 1:
                for i in range(len(trajectory) - 1):
                    pt1 = tuple(trajectory[i].astype(int))
                    pt2 = tuple(trajectory[i + 1].astype(int))
                    alpha = (i + 1) / len(trajectory)  # Fade older points
                    traj_color = tuple(int(c * alpha) for c in base_color)
                    cv2.line(vis_img, pt1, pt2, traj_color, 2)
        
        # Prepare track information text
        conf = getattr(track, 'conf', 0.0)
        quality = getattr(track, 'quality_score', conf)
        
        info_lines = [
            f"ID:{track_id}",
            f"Q:{quality:.2f}",
            f"C:{conf:.2f}"
        ]
        
        if occlusion_level > 0.1:
            info_lines.append(f"Occ:{occlusion_level:.1%}")
        
        # Draw track information with background
        text_y_offset = 0
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_img, 
                         (x1, y1 - 20 - text_y_offset), 
                         (x1 + text_size[0] + 5, y1 - text_y_offset), 
                         base_color, -1)
            cv2.putText(vis_img, line, 
                       (x1 + 2, y1 - 5 - text_y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            text_y_offset += 20
        
        # Draw occlusion progress bar
        if occlusion_level > 0.1:
            bar_width = int((x2 - x1) * occlusion_level)
            cv2.rectangle(vis_img, (x1, y2 - 5), (x1 + bar_width, y2), 
                         occlusion_colors[occlusion_status], -1)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_img, "Occlusion Status:", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    legend_items = [
        ("Visible", occlusion_colors['visible']),
        ("Partial", occlusion_colors['partial']),
        ("Occluded", occlusion_colors['occluded'])
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + 20 + i * 20
        cv2.rectangle(vis_img, (10, y_pos), (30, y_pos + 15), color, -1)
        cv2.putText(vis_img, label, (35, y_pos + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img


# Legacy compatibility
class MetricsPlotter(EnhancedMetricsPlotter):
    """Legacy compatibility class - redirects to EnhancedMetricsPlotter"""
    pass


def create_tracking_dashboard(metrics_data: Dict, 
                            occlusion_data: Dict,
                            output_dir: str = 'dashboard',
                            title: str = 'Enhanced Tracking Dashboard'):
    """
    Create a comprehensive tracking dashboard with all visualizations.
    
    Parameters
    ----------
    metrics_data : dict
        Dictionary containing all tracking metrics
    occlusion_data : dict
        Dictionary containing occlusion analysis data
    output_dir : str
        Directory to save dashboard components
    title : str
        Title for the dashboard
    """
    plotter = EnhancedMetricsPlotter(output_dir)
    
    # Generate all plots
    print("📊 Generating tracking performance dashboard...")
    
    if 'radar_data' in metrics_data:
        plotter.plot_radar_chart(
            metrics_data['radar_data'], 
            metrics_data.get('radar_labels', []),
            title=f"{title} - Overall Performance"
        )
    
    if occlusion_data:
        plotter.plot_occlusion_analysis(occlusion_data)
    
    if 'trajectories' in metrics_data:
        plotter.plot_track_trajectories(
            metrics_data['trajectories'],
            metrics_data.get('image_size', (1920, 1080))
        )
    
    if 'comparison_data' in metrics_data:
        plotter.plot_performance_comparison(
            metrics_data['comparison_data'],
            metrics_data.get('comparison_metrics', ['MOTA', 'IDF1', 'HOTA'])
        )
    
    if 'realtime_metrics' in metrics_data:
        plotter.plot_real_time_metrics(metrics_data['realtime_metrics'])
    
    if 'quality_matrix' in metrics_data:
        plotter.plot_quality_heatmap(
            metrics_data['quality_matrix'],
            metrics_data.get('track_ids', []),
            metrics_data.get('frame_range', (0, 100))
        )
    
    print(f"✅ Dashboard generated in: {output_dir}")
    return output_dir
