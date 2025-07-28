# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Enhanced StrongSort Utilities

This module provides comprehensive utilities for StrongSort tracker analysis,
performance optimization, visualization, and debugging.
"""

import json
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class TrackingMetrics:
    """Data class for storing tracking performance metrics"""
    frame_id: int
    total_detections: int
    total_tracks: int
    confirmed_tracks: int
    tentative_tracks: int
    matched_pairs: int
    unmatched_detections: int
    unmatched_tracks: int
    avg_track_quality: float
    avg_track_confidence: float
    id_switches: int
    new_tracks: int
    lost_tracks: int
    processing_time_ms: float


class StrongSortAnalyzer:
    """
    Comprehensive analyzer for StrongSort tracker performance.
    
    Provides detailed analysis of tracking performance including:
    - ID preservation analysis
    - Matching quality assessment
    - Performance bottleneck identification
    - Parameter optimization suggestions
    """
    
    def __init__(self, history_length: int = 1000):
        self.history_length = history_length
        self.reset()
    
    def reset(self):
        """Reset all tracking statistics"""
        self.metrics_history: List[TrackingMetrics] = []
        self.track_lifetimes: Dict[int, List[int]] = defaultdict(list)
        self.id_switches: List[Tuple[int, int, int]] = []  # (frame, old_id, new_id)
        self.matching_costs: List[float] = []
        self.feature_similarities: List[float] = []
        self.motion_consistencies: List[float] = []
        self.processing_times: Dict[str, List[float]] = defaultdict(list)
        
        # Advanced analytics
        self.track_quality_evolution: Dict[int, List[float]] = defaultdict(list)
        self.appearance_consistency_stats: List[float] = []
        self.detection_quality_stats: List[float] = []
        self.scene_complexity_metrics: List[Dict] = []
    
    def update_metrics(self, tracker, detections, matches, unmatched_tracks, 
                      unmatched_detections, processing_time: float, frame_id: int):
        """Update tracking metrics for current frame"""
        
        # Basic metrics
        total_detections = len(detections)
        total_tracks = len(tracker.tracks)
        confirmed_tracks = len([t for t in tracker.tracks if t.is_confirmed()])
        tentative_tracks = total_tracks - confirmed_tracks
        matched_pairs = len(matches)
        
        # Quality metrics
        if tracker.tracks:
            avg_track_quality = np.mean([getattr(t, 'quality_score', 0.5) for t in tracker.tracks])
            avg_track_confidence = np.mean([t.conf for t in tracker.tracks])
        else:
            avg_track_quality = 0.0
            avg_track_confidence = 0.0
        
        # Detect ID switches (simplified)
        id_switches = self._detect_id_switches(tracker.tracks, frame_id)
        
        # Count new and lost tracks
        new_tracks = len([t for t in tracker.tracks if t.age == 1])
        lost_tracks = len(unmatched_tracks)
        
        # Create metrics object
        metrics = TrackingMetrics(
            frame_id=frame_id,
            total_detections=total_detections,
            total_tracks=total_tracks,
            confirmed_tracks=confirmed_tracks,
            tentative_tracks=tentative_tracks,
            matched_pairs=matched_pairs,
            unmatched_detections=len(unmatched_detections),
            unmatched_tracks=len(unmatched_tracks),
            avg_track_quality=avg_track_quality,
            avg_track_confidence=avg_track_confidence,
            id_switches=len(id_switches),
            new_tracks=new_tracks,
            lost_tracks=lost_tracks,
            processing_time_ms=processing_time * 1000
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_length:
            self.metrics_history.pop(0)
        
        # Update track lifetimes
        for track in tracker.tracks:
            self.track_lifetimes[track.id].append(frame_id)
        
        # Store advanced analytics
        self._update_advanced_analytics(tracker, detections)
    
    def _detect_id_switches(self, tracks, frame_id: int) -> List[Tuple[int, int]]:
        """Detect potential ID switches (simplified implementation)"""
        # This is a simplified version - in practice, you'd need more sophisticated logic
        id_switches = []
        # Implementation would compare track positions and features across frames
        return id_switches
    
    def _update_advanced_analytics(self, tracker, detections):
        """Update advanced analytics metrics"""
        
        # Track quality evolution
        for track in tracker.tracks:
            quality = getattr(track, 'quality_score', 0.5)
            self.track_quality_evolution[track.id].append(quality)
        
        # Appearance consistency
        for track in tracker.tracks:
            consistency = getattr(track, 'appearance_consistency', 1.0)
            self.appearance_consistency_stats.append(consistency)
        
        # Detection quality
        for detection in detections:
            quality = getattr(detection, 'quality_score', detection.conf)
            self.detection_quality_stats.append(quality)
        
        # Scene complexity
        complexity = {
            'detection_density': len(detections),
            'track_density': len(tracker.tracks),
            'avg_detection_conf': np.mean([d.conf for d in detections]) if detections else 0,
            'motion_variance': self._calculate_motion_variance(tracker.tracks)
        }
        self.scene_complexity_metrics.append(complexity)
    
    def _calculate_motion_variance(self, tracks) -> float:
        """Calculate motion variance across all tracks"""
        velocities = []
        for track in tracks:
            if hasattr(track, 'velocity_history') and track.velocity_history:
                velocities.extend([np.linalg.norm(v) for v in track.velocity_history])
        
        return np.var(velocities) if velocities else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 frames
        
        summary = {
            'overall': {
                'total_frames': len(self.metrics_history),
                'avg_processing_time_ms': np.mean([m.processing_time_ms for m in recent_metrics]),
                'avg_match_rate': np.mean([m.matched_pairs / max(m.total_detections, 1) for m in recent_metrics]),
                'avg_track_quality': np.mean([m.avg_track_quality for m in recent_metrics]),
                'total_id_switches': sum([m.id_switches for m in recent_metrics]),
            },
            'detection_analysis': {
                'avg_detections_per_frame': np.mean([m.total_detections for m in recent_metrics]),
                'detection_variance': np.var([m.total_detections for m in recent_metrics]),
                'avg_detection_quality': np.mean(self.detection_quality_stats[-1000:]) if self.detection_quality_stats else 0,
            },
            'track_analysis': {
                'avg_tracks_per_frame': np.mean([m.total_tracks for m in recent_metrics]),
                'avg_track_lifetime': self._calculate_avg_track_lifetime(),
                'track_stability_score': self._calculate_track_stability_score(),
                'appearance_consistency': np.mean(self.appearance_consistency_stats[-1000:]) if self.appearance_consistency_stats else 0,
            },
            'matching_analysis': {
                'avg_unmatched_detections': np.mean([m.unmatched_detections for m in recent_metrics]),
                'avg_unmatched_tracks': np.mean([m.unmatched_tracks for m in recent_metrics]),
                'match_efficiency': self._calculate_match_efficiency(),
            },
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _calculate_avg_track_lifetime(self) -> float:
        """Calculate average track lifetime"""
        lifetimes = []
        for track_id, frames in self.track_lifetimes.items():
            if len(frames) > 1:
                lifetime = max(frames) - min(frames) + 1
                lifetimes.append(lifetime)
        
        return np.mean(lifetimes) if lifetimes else 0.0
    
    def _calculate_track_stability_score(self) -> float:
        """Calculate overall track stability score"""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = self.metrics_history[-50:]
        
        # Stability based on consistent number of tracks and low ID switches
        track_count_variance = np.var([m.total_tracks for m in recent_metrics])
        avg_id_switches = np.mean([m.id_switches for m in recent_metrics])
        
        # Normalize to 0-1 scale
        stability = 1.0 / (1.0 + track_count_variance + avg_id_switches)
        
        return stability
    
    def _calculate_match_efficiency(self) -> float:
        """Calculate matching efficiency score"""
        if not self.metrics_history:
            return 0.0
        
        recent_metrics = self.metrics_history[-100:]
        
        total_possible_matches = sum([min(m.total_detections, m.total_tracks) for m in recent_metrics])
        actual_matches = sum([m.matched_pairs for m in recent_metrics])
        
        return actual_matches / max(total_possible_matches, 1)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate parameter tuning recommendations based on analysis"""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        recent_metrics = self.metrics_history[-100:]
        
        # Analysis-based recommendations
        avg_match_rate = np.mean([m.matched_pairs / max(m.total_detections, 1) for m in recent_metrics])
        avg_id_switches = np.mean([m.id_switches for m in recent_metrics])
        avg_processing_time = np.mean([m.processing_time_ms for m in recent_metrics])
        
        if avg_match_rate < 0.7:
            recommendations.append("Low match rate detected. Consider increasing max_cos_dist or max_iou_dist.")
        
        if avg_id_switches > 0.1:
            recommendations.append("High ID switching rate. Consider decreasing max_cos_dist or increasing nn_budget.")
        
        if avg_processing_time > 50:
            recommendations.append("High processing time. Consider reducing nn_budget or using faster ReID model.")
        
        if len(recent_metrics) > 0:
            track_variance = np.var([m.total_tracks for m in recent_metrics])
            if track_variance > 10:
                recommendations.append("High track count variance. Consider adjusting max_age and n_init parameters.")
        
        return recommendations
    
    def plot_performance_metrics(self, save_path: Optional[str] = None):
        """Plot comprehensive performance metrics"""
        if not self.metrics_history:
            print("No metrics data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('StrongSort Performance Analysis', fontsize=16)
        
        frames = [m.frame_id for m in self.metrics_history]
        
        # Plot 1: Detection and Track counts
        axes[0, 0].plot(frames, [m.total_detections for m in self.metrics_history], 'b-', label='Detections')
        axes[0, 0].plot(frames, [m.total_tracks for m in self.metrics_history], 'r-', label='Tracks')
        axes[0, 0].plot(frames, [m.confirmed_tracks for m in self.metrics_history], 'g-', label='Confirmed')
        axes[0, 0].set_title('Detection and Track Counts')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Matching performance
        match_rates = [m.matched_pairs / max(m.total_detections, 1) for m in self.metrics_history]
        axes[0, 1].plot(frames, match_rates, 'purple')
        axes[0, 1].set_title('Match Rate')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Match Rate')
        axes[0, 1].grid(True)
        
        # Plot 3: Quality metrics
        axes[0, 2].plot(frames, [m.avg_track_quality for m in self.metrics_history], 'orange', label='Track Quality')
        axes[0, 2].plot(frames, [m.avg_track_confidence for m in self.metrics_history], 'cyan', label='Confidence')
        axes[0, 2].set_title('Quality Metrics')
        axes[0, 2].set_xlabel('Frame')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Processing time
        axes[1, 0].plot(frames, [m.processing_time_ms for m in self.metrics_history], 'red')
        axes[1, 0].set_title('Processing Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True)
        
        # Plot 5: ID switches and track changes
        axes[1, 1].plot(frames, [m.id_switches for m in self.metrics_history], 'brown', label='ID Switches')
        axes[1, 1].plot(frames, [m.new_tracks for m in self.metrics_history], 'green', label='New Tracks')
        axes[1, 1].plot(frames, [m.lost_tracks for m in self.metrics_history], 'red', label='Lost Tracks')
        axes[1, 1].set_title('Track Changes')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: Unmatched detections and tracks
        axes[1, 2].plot(frames, [m.unmatched_detections for m in self.metrics_history], 'orange', label='Unmatched Detections')
        axes[1, 2].plot(frames, [m.unmatched_tracks for m in self.metrics_history], 'purple', label='Unmatched Tracks')
        axes[1, 2].set_title('Unmatched Items')
        axes[1, 2].set_xlabel('Frame')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plot saved to {save_path}")
        else:
            plt.show()
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        data = {
            'metrics_history': [
                {
                    'frame_id': m.frame_id,
                    'total_detections': m.total_detections,
                    'total_tracks': m.total_tracks,
                    'confirmed_tracks': m.confirmed_tracks,
                    'matched_pairs': m.matched_pairs,
                    'avg_track_quality': m.avg_track_quality,
                    'processing_time_ms': m.processing_time_ms,
                } for m in self.metrics_history
            ],
            'summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Metrics exported to {filepath}")


class ParameterTuner:
    """
    Automated parameter tuning for StrongSort tracker.
    
    Uses performance metrics to suggest optimal parameters for different scenarios.
    """
    
    def __init__(self):
        self.parameter_ranges = {
            'max_cos_dist': (0.1, 0.4),
            'max_iou_dist': (0.5, 0.95),
            'max_age': (10, 100),
            'n_init': (1, 5),
            'nn_budget': (50, 300),
            'mc_lambda': (0.9, 0.999),
            'ema_alpha': (0.7, 0.95),
            'conf_thresh_high': (0.6, 0.9),
            'conf_thresh_low': (0.2, 0.5),
        }
    
    def suggest_parameters(self, performance_summary: Dict) -> Dict[str, float]:
        """Suggest optimal parameters based on performance analysis"""
        suggestions = {}
        
        if not performance_summary:
            return self._get_default_parameters()
        
        overall = performance_summary.get('overall', {})
        matching = performance_summary.get('matching_analysis', {})
        
        # Adjust based on match rate
        match_efficiency = matching.get('match_efficiency', 0.5)
        if match_efficiency < 0.6:
            suggestions['max_cos_dist'] = 0.3  # Increase for more lenient matching
            suggestions['max_iou_dist'] = 0.8
        elif match_efficiency > 0.9:
            suggestions['max_cos_dist'] = 0.15  # Decrease for stricter matching
            suggestions['max_iou_dist'] = 0.7
        
        # Adjust based on ID switches
        id_switches = overall.get('total_id_switches', 0)
        avg_frames = overall.get('total_frames', 100)
        id_switch_rate = id_switches / max(avg_frames, 1)
        
        if id_switch_rate > 0.05:  # High ID switch rate
            suggestions['nn_budget'] = 200  # Increase memory
            suggestions['max_cos_dist'] = 0.15  # Stricter appearance matching
            suggestions['ema_alpha'] = 0.85  # Less aggressive feature updating
        
        # Adjust based on processing time
        processing_time = overall.get('avg_processing_time_ms', 25)
        if processing_time > 50:
            suggestions['nn_budget'] = 100  # Reduce for speed
        elif processing_time < 15:
            suggestions['nn_budget'] = 200  # Can afford more memory
        
        # Adjust based on track stability
        track_analysis = performance_summary.get('track_analysis', {})
        track_lifetime = track_analysis.get('avg_track_lifetime', 10)
        
        if track_lifetime < 5:  # Short track lifetimes
            suggestions['max_age'] = 50  # Keep tracks longer
            suggestions['n_init'] = 2  # Faster confirmation
        elif track_lifetime > 50:  # Very long tracks
            suggestions['max_age'] = 30  # Normal deletion
            suggestions['n_init'] = 3  # More conservative confirmation
        
        return suggestions
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default enhanced parameters"""
        return {
            'max_cos_dist': 0.15,
            'max_iou_dist': 0.7,
            'max_age': 50,
            'n_init': 2,
            'nn_budget': 150,
            'mc_lambda': 0.995,
            'ema_alpha': 0.9,
            'conf_thresh_high': 0.7,
            'conf_thresh_low': 0.3,
        }


class QualityAssessor:
    """
    Assess tracking quality in real-time and provide feedback.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.recent_metrics = deque(maxlen=window_size)
    
    def assess_frame_quality(self, tracker, detections, matches, frame_id: int) -> Dict[str, float]:
        """Assess quality of tracking for current frame"""
        
        quality_scores = {}
        
        # Match quality
        if detections:
            match_rate = len(matches) / len(detections)
            quality_scores['match_quality'] = match_rate
        else:
            quality_scores['match_quality'] = 1.0
        
        # Track quality
        if tracker.tracks:
            track_qualities = [getattr(t, 'quality_score', 0.5) for t in tracker.tracks]
            quality_scores['track_quality'] = np.mean(track_qualities)
            
            # Stability assessment
            stable_tracks = len([t for t in tracker.tracks if t.hits > 5])
            quality_scores['stability'] = stable_tracks / len(tracker.tracks)
        else:
            quality_scores['track_quality'] = 0.0
            quality_scores['stability'] = 0.0
        
        # Detection quality
        if detections:
            detection_qualities = [getattr(d, 'quality_score', d.conf) for d in detections]
            quality_scores['detection_quality'] = np.mean(detection_qualities)
        else:
            quality_scores['detection_quality'] = 0.0
        
        # Overall quality
        quality_scores['overall'] = np.mean([
            quality_scores['match_quality'],
            quality_scores['track_quality'],
            quality_scores['stability'],
            quality_scores['detection_quality']
        ])
        
        self.recent_metrics.append(quality_scores)
        
        return quality_scores
    
    def get_quality_trends(self) -> Dict[str, float]:
        """Get quality trends over recent frames"""
        if len(self.recent_metrics) < 2:
            return {}
        
        trends = {}
        metrics_array = np.array([list(m.values()) for m in self.recent_metrics])
        metric_names = list(self.recent_metrics[0].keys())
        
        for i, name in enumerate(metric_names):
            values = metrics_array[:, i]
            if len(values) > 5:
                # Simple trend calculation
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[f'{name}_trend'] = trend
        
        return trends
    
    def get_quality_alerts(self) -> List[str]:
        """Get quality alerts based on recent performance"""
        alerts = []
        
        if len(self.recent_metrics) < 10:
            return alerts
        
        recent_avg = {k: np.mean([m[k] for m in self.recent_metrics]) for k in self.recent_metrics[0].keys()}
        
        if recent_avg['match_quality'] < 0.5:
            alerts.append("Low match rate detected - consider adjusting matching thresholds")
        
        if recent_avg['track_quality'] < 0.4:
            alerts.append("Poor track quality - check ReID model and feature extraction")
        
        if recent_avg['stability'] < 0.3:
            alerts.append("Low track stability - many short-lived tracks detected")
        
        if recent_avg['detection_quality'] < 0.4:
            alerts.append("Poor detection quality - check detection model confidence")
        
        return alerts


def visualize_tracking_state(img, tracker, detections=None, save_path=None):
    """
    Visualize current tracking state with enhanced information display.
    """
    if not CV2_AVAILABLE:
        print("OpenCV not available for visualization")
        return img
    
    vis_img = img.copy()
    
    # Color palette for tracks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    # Draw tracks
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = track.to_tlbr()
        color = colors[track.id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Track info
        quality = getattr(track, 'quality_score', 0.5)
        info_text = f"ID:{track.id} Q:{quality:.2f} C:{track.conf:.2f}"
        
        # Background for text
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis_img, (int(x1), int(y1-25)), (int(x1 + text_size[0]), int(y1)), color, -1)
        
        # Text
        cv2.putText(vis_img, info_text, (int(x1), int(y1-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Track trajectory (if available)
        if hasattr(track, 'position_history') and len(track.position_history) > 1:
            points = [(int(p[0]), int(p[1])) for p in track.position_history[-10:]]
            for j in range(1, len(points)):
                cv2.line(vis_img, points[j-1], points[j], color, 1)
    
    # Draw unmatched detections
    if detections:
        for detection in detections:
            x, y, w, h = detection.tlwh
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # Draw as dashed rectangle (simplified)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1)
            cv2.putText(vis_img, f"Det:{detection.conf:.2f}", 
                       (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Add overall statistics
    stats_text = [
        f"Tracks: {len(tracker.tracks)}",
        f"Confirmed: {len([t for t in tracker.tracks if t.is_confirmed()])}",
        f"Next ID: {tracker._next_id}",
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(vis_img, text, (10, 30 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img


def create_tracking_report(analyzer: StrongSortAnalyzer, save_dir: str):
    """
    Create comprehensive tracking performance report.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get performance summary
    summary = analyzer.get_performance_summary()
    
    # Save summary to JSON
    with open(save_path / "tracking_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create performance plots
    analyzer.plot_performance_metrics(str(save_path / "performance_plots.png"))
    
    # Generate recommendations
    tuner = ParameterTuner()
    recommendations = tuner.suggest_parameters(summary)
    
    with open(save_path / "parameter_recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    # Create markdown report
    report_content = f"""
# StrongSort Tracking Performance Report

## Executive Summary

- **Total Frames Processed**: {summary.get('overall', {}).get('total_frames', 0)}
- **Average Processing Time**: {summary.get('overall', {}).get('avg_processing_time_ms', 0):.2f} ms
- **Average Match Rate**: {summary.get('overall', {}).get('avg_match_rate', 0):.2%}
- **Average Track Quality**: {summary.get('overall', {}).get('avg_track_quality', 0):.2f}

## Performance Metrics

### Detection Analysis
- Average Detections per Frame: {summary.get('detection_analysis', {}).get('avg_detections_per_frame', 0):.1f}
- Detection Quality Score: {summary.get('detection_analysis', {}).get('avg_detection_quality', 0):.2f}

### Track Analysis
- Average Tracks per Frame: {summary.get('track_analysis', {}).get('avg_tracks_per_frame', 0):.1f}
- Average Track Lifetime: {summary.get('track_analysis', {}).get('avg_track_lifetime', 0):.1f} frames
- Track Stability Score: {summary.get('track_analysis', {}).get('track_stability_score', 0):.2f}

### Matching Analysis
- Match Efficiency: {summary.get('matching_analysis', {}).get('match_efficiency', 0):.2%}
- Average Unmatched Detections: {summary.get('matching_analysis', {}).get('avg_unmatched_detections', 0):.1f}
- Average Unmatched Tracks: {summary.get('matching_analysis', {}).get('avg_unmatched_tracks', 0):.1f}

## Recommendations

"""
    
    for rec in summary.get('recommendations', []):
        report_content += f"- {rec}\n"
    
    report_content += f"""

## Suggested Parameters

```json
{json.dumps(recommendations, indent=2)}
```

---
Generated by StrongSort Enhanced Utilities
"""
    
    with open(save_path / "tracking_report.md", 'w') as f:
        f.write(report_content)
    
    print(f"Comprehensive tracking report saved to {save_path}")


# Utility functions for common tasks

def load_tracking_config(config_path: str) -> Dict:
    """Load tracking configuration from file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            return yaml.safe_load(f)
    raise ValueError("Unsupported config file format")


def save_tracking_config(config: Dict, config_path: str):
    """Save tracking configuration to file"""
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            yaml.dump(config, f, default_flow_style=False)


def benchmark_tracker_performance(tracker, test_data, num_runs: int = 10) -> Dict:
    """Benchmark tracker performance on test data"""
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        # Simulate tracking on test data
        for frame_data in test_data:
            img, detections = frame_data
            tracker.update(detections, img)
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'frames_processed': len(test_data) * num_runs,
        'fps': len(test_data) / (np.mean(times) / 1000)
    } 