# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Enhanced Tracking Metrics Module

Comprehensive evaluation metrics for multi-object tracking with special focus on:
- Occlusion-aware evaluation
- ID preservation analysis
- Real-time performance monitoring
- Quality assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict, deque
from dataclasses import dataclass
import time
from pathlib import Path
import json


@dataclass
class TrackingMetrics:
    """Container for tracking evaluation metrics"""
    # Basic MOT metrics
    mota: float = 0.0           # Multiple Object Tracking Accuracy
    motp: float = 0.0           # Multiple Object Tracking Precision
    idf1: float = 0.0           # ID F1 Score
    hota: float = 0.0           # Higher Order Tracking Accuracy
    
    # ID metrics
    id_switches: int = 0        # Number of ID switches
    id_fragmentations: int = 0  # Number of fragmentations
    id_recovery_rate: float = 0.0  # Rate of successful ID recovery
    
    # Detection metrics
    recall: float = 0.0         # Detection recall
    precision: float = 0.0      # Detection precision
    f1_score: float = 0.0       # Detection F1 score
    
    # Count metrics
    mostly_tracked: int = 0     # Tracks with >80% coverage
    mostly_lost: int = 0        # Tracks with <20% coverage
    partially_tracked: int = 0  # Tracks with 20-80% coverage
    
    # Occlusion-specific metrics
    occlusion_accuracy: float = 0.0      # Accuracy during occlusion
    id_preservation_in_occlusion: float = 0.0  # ID preservation during occlusion
    occlusion_recovery_rate: float = 0.0       # Recovery rate after occlusion
    
    # Performance metrics
    avg_processing_time: float = 0.0     # Average processing time per frame
    fps: float = 0.0                     # Frames per second
    memory_usage: float = 0.0            # Memory usage in MB


class TrackingEvaluator:
    """Enhanced tracking evaluator with occlusion awareness"""
    
    def __init__(self, iou_threshold: float = 0.5, 
                 occlusion_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.occlusion_threshold = occlusion_threshold
        
        # Tracking data
        self.gt_tracks = {}           # Ground truth tracks
        self.pred_tracks = {}         # Predicted tracks
        self.frame_data = {}          # Per-frame data
        
        # Metrics accumulation
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_id_switches = 0
        self.total_fragmentations = 0
        
        # Occlusion tracking
        self.occlusion_events = []
        self.id_switches_in_occlusion = 0
        self.occlusion_frames = set()
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.frame_count = 0
        
        # Track history for analysis
        self.track_history = defaultdict(list)
        self.gt_track_history = defaultdict(list)
        
    def add_frame_data(self, frame_id: int, 
                      gt_detections: List[Dict],
                      pred_detections: List[Dict],
                      processing_time: float = 0.0,
                      occlusion_info: Dict = None):
        """
        Add data for a single frame.
        
        Args:
            frame_id: Frame identifier
            gt_detections: Ground truth detections with 'bbox', 'id', 'occluded' fields
            pred_detections: Predicted detections with 'bbox', 'id', 'conf' fields
            processing_time: Processing time for this frame
            occlusion_info: Additional occlusion information
        """
        self.frame_count += 1
        self.processing_times.append(processing_time)
        
        # Store frame data
        self.frame_data[frame_id] = {
            'gt': gt_detections,
            'pred': pred_detections,
            'occlusion_info': occlusion_info or {}
        }
        
        # Update track histories
        for gt_det in gt_detections:
            track_id = gt_det['id']
            self.gt_track_history[track_id].append({
                'frame': frame_id,
                'bbox': gt_det['bbox'],
                'occluded': gt_det.get('occluded', False)
            })
        
        for pred_det in pred_detections:
            track_id = pred_det['id']
            self.track_history[track_id].append({
                'frame': frame_id,
                'bbox': pred_det['bbox'],
                'conf': pred_det.get('conf', 1.0)
            })
        
        # Detect occlusion frames
        if any(det.get('occluded', False) for det in gt_detections):
            self.occlusion_frames.add(frame_id)
    
    def compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes in [x1, y1, x2, y2] format"""
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def associate_detections(self, gt_detections: List[Dict], 
                           pred_detections: List[Dict]) -> Tuple[List, List, List]:
        """
        Associate ground truth and predicted detections using Hungarian algorithm.
        
        Returns:
            matches: List of (gt_idx, pred_idx) pairs
            unmatched_gt: List of unmatched GT indices
            unmatched_pred: List of unmatched prediction indices
        """
        if not gt_detections or not pred_detections:
            return [], list(range(len(gt_detections))), list(range(len(pred_detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(gt_detections), len(pred_detections)))
        for i, gt_det in enumerate(gt_detections):
            for j, pred_det in enumerate(pred_detections):
                iou_matrix[i, j] = self.compute_iou(gt_det['bbox'], pred_det['bbox'])
        
        # Hungarian assignment
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        for i, j in zip(row_indices, col_indices):
            if iou_matrix[i, j] >= self.iou_threshold:
                matches.append((i, j))
        
        # Find unmatched
        matched_gt = set([m[0] for m in matches])
        matched_pred = set([m[1] for m in matches])
        
        unmatched_gt = [i for i in range(len(gt_detections)) if i not in matched_gt]
        unmatched_pred = [j for j in range(len(pred_detections)) if j not in matched_pred]
        
        return matches, unmatched_gt, unmatched_pred
    
    def evaluate_frame(self, frame_id: int) -> Dict:
        """Evaluate a single frame and return metrics"""
        if frame_id not in self.frame_data:
            return {}
        
        data = self.frame_data[frame_id]
        gt_detections = data['gt']
        pred_detections = data['pred']
        
        # Associate detections
        matches, unmatched_gt, unmatched_pred = self.associate_detections(
            gt_detections, pred_detections
        )
        
        # Count metrics
        tp = len(matches)
        fn = len(unmatched_gt)
        fp = len(unmatched_pred)
        
        self.total_tp += tp
        self.total_fn += fn
        self.total_fp += fp
        
        # ID analysis for matched detections
        id_switches_frame = 0
        correct_ids = 0
        
        for gt_idx, pred_idx in matches:
            gt_id = gt_detections[gt_idx]['id']
            pred_id = pred_detections[pred_idx]['id']
            
            if gt_id == pred_id:
                correct_ids += 1
            else:
                id_switches_frame += 1
                self.total_id_switches += 1
                
                # Check if this is during occlusion
                if frame_id in self.occlusion_frames:
                    self.id_switches_in_occlusion += 1
        
        return {
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'id_switches': id_switches_frame,
            'correct_ids': correct_ids,
            'total_detections': len(gt_detections),
            'is_occlusion_frame': frame_id in self.occlusion_frames
        }
    
    def compute_track_metrics(self) -> Dict:
        """Compute track-level metrics (MT, ML, PT, fragmentations)"""
        track_stats = {}
        
        for track_id, history in self.gt_track_history.items():
            total_frames = len(history)
            detected_frames = 0
            
            # Check how many frames this track was detected
            for frame_data in history:
                frame_id = frame_data['frame']
                if frame_id in self.frame_data:
                    # Check if this track was detected in predictions
                    pred_detections = self.frame_data[frame_id]['pred']
                    for pred_det in pred_detections:
                        if pred_det['id'] == track_id:
                            # Additional IoU check
                            iou = self.compute_iou(frame_data['bbox'], pred_det['bbox'])
                            if iou >= self.iou_threshold:
                                detected_frames += 1
                                break
            
            coverage = detected_frames / total_frames if total_frames > 0 else 0
            track_stats[track_id] = {
                'coverage': coverage,
                'total_frames': total_frames,
                'detected_frames': detected_frames
            }
        
        # Classify tracks
        mostly_tracked = sum(1 for stats in track_stats.values() if stats['coverage'] > 0.8)
        mostly_lost = sum(1 for stats in track_stats.values() if stats['coverage'] < 0.2)
        partially_tracked = len(track_stats) - mostly_tracked - mostly_lost
        
        return {
            'mostly_tracked': mostly_tracked,
            'mostly_lost': mostly_lost,
            'partially_tracked': partially_tracked,
            'total_tracks': len(track_stats),
            'track_stats': track_stats
        }
    
    def compute_occlusion_metrics(self) -> Dict:
        """Compute occlusion-specific metrics"""
        if not self.occlusion_frames:
            return {
                'occlusion_accuracy': 1.0,
                'id_preservation_in_occlusion': 1.0,
                'occlusion_recovery_rate': 1.0,
                'total_occlusion_frames': 0
            }
        
        # Evaluate performance during occlusion frames
        occlusion_tp = 0
        occlusion_total = 0
        
        for frame_id in self.occlusion_frames:
            frame_metrics = self.evaluate_frame(frame_id)
            if frame_metrics:
                occlusion_tp += frame_metrics.get('correct_ids', 0)
                occlusion_total += frame_metrics.get('total_detections', 0)
        
        occlusion_accuracy = occlusion_tp / occlusion_total if occlusion_total > 0 else 1.0
        
        # ID preservation during occlusion
        total_id_switches_in_occlusion = self.id_switches_in_occlusion
        total_matches_in_occlusion = occlusion_tp
        
        id_preservation = 1.0 - (total_id_switches_in_occlusion / max(total_matches_in_occlusion, 1))
        
        # Recovery rate (simplified - tracks that continue after occlusion)
        recovery_count = 0
        total_recovery_opportunities = 0
        
        for track_id, history in self.gt_track_history.items():
            in_occlusion = False
            for i, frame_data in enumerate(history):
                if frame_data.get('occluded', False):
                    in_occlusion = True
                elif in_occlusion:
                    # Just came out of occlusion
                    total_recovery_opportunities += 1
                    # Check if track continued
                    if i < len(history) - 1:  # Not the last frame
                        recovery_count += 1
                    in_occlusion = False
        
        recovery_rate = recovery_count / max(total_recovery_opportunities, 1)
        
        return {
            'occlusion_accuracy': occlusion_accuracy,
            'id_preservation_in_occlusion': id_preservation,
            'occlusion_recovery_rate': recovery_rate,
            'total_occlusion_frames': len(self.occlusion_frames),
            'id_switches_in_occlusion': total_id_switches_in_occlusion
        }
    
    def compute_final_metrics(self) -> TrackingMetrics:
        """Compute final comprehensive metrics"""
        # Basic detection metrics
        total_gt = self.total_tp + self.total_fn
        total_pred = self.total_tp + self.total_fp
        
        recall = self.total_tp / max(total_gt, 1)
        precision = self.total_tp / max(total_pred, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        # MOTA computation
        mota = 1 - (self.total_fn + self.total_fp + self.total_id_switches) / max(total_gt, 1)
        
        # MOTP computation (simplified)
        motp = precision  # Simplified MOTP as precision
        
        # Track-level metrics
        track_metrics = self.compute_track_metrics()
        
        # Occlusion metrics
        occlusion_metrics = self.compute_occlusion_metrics()
        
        # Performance metrics
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        
        # ID metrics (simplified)
        idf1 = f1_score  # Simplified IDF1
        hota = (recall + precision) / 2  # Simplified HOTA
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=idf1,
            hota=hota,
            id_switches=self.total_id_switches,
            id_fragmentations=0,  # TODO: Implement fragmentation detection
            id_recovery_rate=occlusion_metrics['occlusion_recovery_rate'],
            recall=recall,
            precision=precision,
            f1_score=f1_score,
            mostly_tracked=track_metrics['mostly_tracked'],
            mostly_lost=track_metrics['mostly_lost'],
            partially_tracked=track_metrics['partially_tracked'],
            occlusion_accuracy=occlusion_metrics['occlusion_accuracy'],
            id_preservation_in_occlusion=occlusion_metrics['id_preservation_in_occlusion'],
            occlusion_recovery_rate=occlusion_metrics['occlusion_recovery_rate'],
            avg_processing_time=avg_processing_time,
            fps=fps,
            memory_usage=0.0  # TODO: Implement memory monitoring
        )
    
    def get_detailed_report(self) -> Dict:
        """Generate detailed evaluation report"""
        metrics = self.compute_final_metrics()
        track_metrics = self.compute_track_metrics()
        occlusion_metrics = self.compute_occlusion_metrics()
        
        return {
            'summary_metrics': {
                'MOTA': metrics.mota,
                'MOTP': metrics.motp,
                'IDF1': metrics.idf1,
                'HOTA': metrics.hota,
                'Recall': metrics.recall,
                'Precision': metrics.precision,
                'F1-Score': metrics.f1_score
            },
            'id_metrics': {
                'ID_Switches': metrics.id_switches,
                'ID_Recovery_Rate': metrics.id_recovery_rate,
                'ID_Preservation_in_Occlusion': metrics.id_preservation_in_occlusion
            },
            'track_metrics': {
                'Mostly_Tracked': metrics.mostly_tracked,
                'Mostly_Lost': metrics.mostly_lost,
                'Partially_Tracked': metrics.partially_tracked,
                'Total_Tracks': track_metrics['total_tracks']
            },
            'occlusion_metrics': {
                'Occlusion_Accuracy': metrics.occlusion_accuracy,
                'Occlusion_Recovery_Rate': metrics.occlusion_recovery_rate,
                'Total_Occlusion_Frames': occlusion_metrics['total_occlusion_frames'],
                'ID_Switches_in_Occlusion': occlusion_metrics['id_switches_in_occlusion']
            },
            'performance_metrics': {
                'Average_Processing_Time_ms': metrics.avg_processing_time * 1000,
                'FPS': metrics.fps,
                'Total_Frames': self.frame_count
            },
            'count_metrics': {
                'Total_TP': self.total_tp,
                'Total_FP': self.total_fp,
                'Total_FN': self.total_fn
            }
        }
    
    def save_report(self, output_path: str):
        """Save evaluation report to file"""
        report = self.get_detailed_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Evaluation report saved to: {output_path}")


class RealTimeMetricsMonitor:
    """Real-time tracking metrics monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Sliding windows for metrics
        self.processing_times = deque(maxlen=window_size)
        self.match_rates = deque(maxlen=window_size)
        self.id_switch_counts = deque(maxlen=window_size)
        self.detection_counts = deque(maxlen=window_size)
        self.track_counts = deque(maxlen=window_size)
        
        # Cumulative counters
        self.total_frames = 0
        self.total_id_switches = 0
        self.total_tracks_created = 0
        
        # Quality indicators
        self.quality_alerts = []
        
    def update(self, frame_metrics: Dict):
        """Update with metrics from a single frame"""
        self.total_frames += 1
        
        # Update sliding windows
        self.processing_times.append(frame_metrics.get('processing_time', 0.0))
        self.match_rates.append(frame_metrics.get('match_rate', 0.0))
        self.id_switch_counts.append(frame_metrics.get('id_switches', 0))
        self.detection_counts.append(frame_metrics.get('detection_count', 0))
        self.track_counts.append(frame_metrics.get('track_count', 0))
        
        # Update cumulative counters
        self.total_id_switches += frame_metrics.get('id_switches', 0)
        self.total_tracks_created += frame_metrics.get('new_tracks', 0)
        
        # Check for quality alerts
        self._check_quality_alerts(frame_metrics)
    
    def _check_quality_alerts(self, frame_metrics: Dict):
        """Check for quality issues and generate alerts"""
        current_time = time.time()
        
        # High processing time alert
        if frame_metrics.get('processing_time', 0) > 0.1:  # >100ms
            self.quality_alerts.append({
                'type': 'HIGH_PROCESSING_TIME',
                'value': frame_metrics['processing_time'],
                'timestamp': current_time,
                'frame': self.total_frames
            })
        
        # Low match rate alert
        if frame_metrics.get('match_rate', 1.0) < 0.5:
            self.quality_alerts.append({
                'type': 'LOW_MATCH_RATE',
                'value': frame_metrics['match_rate'],
                'timestamp': current_time,
                'frame': self.total_frames
            })
        
        # High ID switch rate alert
        recent_id_switches = sum(list(self.id_switch_counts)[-10:])  # Last 10 frames
        if recent_id_switches > 5:
            self.quality_alerts.append({
                'type': 'HIGH_ID_SWITCH_RATE',
                'value': recent_id_switches,
                'timestamp': current_time,
                'frame': self.total_frames
            })
        
        # Keep only recent alerts (last 1000 frames)
        max_alerts = 1000
        if len(self.quality_alerts) > max_alerts:
            self.quality_alerts = self.quality_alerts[-max_alerts:]
    
    def get_current_metrics(self) -> Dict:
        """Get current real-time metrics"""
        if not self.processing_times:
            return {}
        
        return {
            'fps': 1.0 / max(np.mean(self.processing_times), 1e-6),
            'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
            'avg_match_rate': np.mean(self.match_rates) if self.match_rates else 0.0,
            'avg_detection_count': np.mean(self.detection_counts) if self.detection_counts else 0.0,
            'avg_track_count': np.mean(self.track_counts) if self.track_counts else 0.0,
            'id_switch_rate': self.total_id_switches / max(self.total_frames, 1),
            'total_frames': self.total_frames,
            'total_id_switches': self.total_id_switches,
            'recent_alerts': len([a for a in self.quality_alerts 
                                if time.time() - a['timestamp'] < 60])  # Last minute
        }
    
    def get_quality_alerts(self, max_age_seconds: float = 300) -> List[Dict]:
        """Get recent quality alerts"""
        current_time = time.time()
        return [alert for alert in self.quality_alerts 
                if current_time - alert['timestamp'] < max_age_seconds]
    
    def reset(self):
        """Reset all metrics"""
        self.processing_times.clear()
        self.match_rates.clear()
        self.id_switch_counts.clear()
        self.detection_counts.clear()
        self.track_counts.clear()
        self.total_frames = 0
        self.total_id_switches = 0
        self.total_tracks_created = 0
        self.quality_alerts.clear()


def compute_mot_metrics(gt_file: str, pred_file: str, 
                       iou_threshold: float = 0.5) -> TrackingMetrics:
    """
    Compute MOT metrics from ground truth and prediction files.
    
    Args:
        gt_file: Path to ground truth file
        pred_file: Path to predictions file
        iou_threshold: IoU threshold for association
        
    Returns:
        TrackingMetrics object with computed metrics
    """
    evaluator = TrackingEvaluator(iou_threshold=iou_threshold)
    
    # Load ground truth
    gt_data = defaultdict(list)
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 9:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                class_id = int(parts[7])
                visibility = float(parts[8])
                
                # Convert to xyxy format
                bbox = [x, y, x + w, y + h]
                gt_data[frame_id].append({
                    'id': track_id,
                    'bbox': bbox,
                    'conf': conf,
                    'class': class_id,
                    'occluded': visibility < 0.5
                })
    
    # Load predictions
    pred_data = defaultdict(list)
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                
                # Convert to xyxy format
                bbox = [x, y, x + w, y + h]
                pred_data[frame_id].append({
                    'id': track_id,
                    'bbox': bbox,
                    'conf': conf
                })
    
    # Evaluate each frame
    all_frames = set(gt_data.keys()) | set(pred_data.keys())
    for frame_id in sorted(all_frames):
        evaluator.add_frame_data(
            frame_id=frame_id,
            gt_detections=gt_data.get(frame_id, []),
            pred_detections=pred_data.get(frame_id, [])
        )
    
    return evaluator.compute_final_metrics()


def print_metrics_summary(metrics: TrackingMetrics, title: str = "Tracking Metrics"):
    """Print a formatted summary of tracking metrics"""
    print(f"\n{title}")
    print("=" * len(title))
    
    print(f"📊 Overall Performance:")
    print(f"   MOTA: {metrics.mota:.3f}")
    print(f"   MOTP: {metrics.motp:.3f}")
    print(f"   IDF1: {metrics.idf1:.3f}")
    print(f"   HOTA: {metrics.hota:.3f}")
    
    print(f"\n🎯 Detection Metrics:")
    print(f"   Recall: {metrics.recall:.3f}")
    print(f"   Precision: {metrics.precision:.3f}")
    print(f"   F1-Score: {metrics.f1_score:.3f}")
    
    print(f"\n🆔 ID Metrics:")
    print(f"   ID Switches: {metrics.id_switches}")
    print(f"   ID Recovery Rate: {metrics.id_recovery_rate:.3f}")
    print(f"   ID Preservation in Occlusion: {metrics.id_preservation_in_occlusion:.3f}")
    
    print(f"\n📈 Track Metrics:")
    print(f"   Mostly Tracked: {metrics.mostly_tracked}")
    print(f"   Partially Tracked: {metrics.partially_tracked}")
    print(f"   Mostly Lost: {metrics.mostly_lost}")
    
    print(f"\n🛡️ Occlusion Metrics:")
    print(f"   Occlusion Accuracy: {metrics.occlusion_accuracy:.3f}")
    print(f"   Occlusion Recovery Rate: {metrics.occlusion_recovery_rate:.3f}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"   Average Processing Time: {metrics.avg_processing_time*1000:.1f} ms")
    print(f"   FPS: {metrics.fps:.1f}")
    print("") 