#!/usr/bin/env python3
"""
Enhanced Utilities Demonstration Script

This script demonstrates the comprehensive enhanced utilities for BoxMOT including:
- Occlusion-aware tracking capabilities
- Advanced visualization and plotting
- Real-time metrics monitoring
- Quality assessment and parameter optimization
- Enhanced association and matching functions
- Comprehensive tracking evaluation

Author: Enhanced BoxMOT Team
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add boxmot to path
sys.path.append(str(Path(__file__).parent.parent))

from boxmot.utils import (
    print_utils_status,
    get_enhanced_tracker_config,
    create_enhanced_tracker,
    ENHANCED_UTILS_AVAILABLE
)

if ENHANCED_UTILS_AVAILABLE:
    from boxmot.utils import (
        # Core analysis tools
        StrongSortAnalyzer, ParameterTuner, QualityAssessor,
        # Occlusion handling
        OcclusionAwareTracker, OverlapAnalyzer, OcclusionStateManager,
        detect_crowd_situations, compute_crowd_density,
        # Visualization
        EnhancedMetricsPlotter, visualize_tracking_frame, create_tracking_dashboard,
        # Metrics and evaluation
        TrackingEvaluator, RealTimeMetricsMonitor, compute_mot_metrics, print_metrics_summary,
        # Enhanced operations
        compute_box_overlap, compute_motion_vector, smooth_box_trajectory,
        compute_box_stability, detect_box_anomalies, interpolate_missing_boxes
    )

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced BoxMOT Utilities Demonstration')
    
    parser.add_argument('--demo-type', type=str, 
                       choices=['all', 'occlusion', 'visualization', 'metrics', 'operations'],
                       default='all', help='Type of demo to run')
    parser.add_argument('--data-path', type=str, default='assets/MOT17-mini/train/MOT17-02-FRCNN',
                       help='Path to MOT sequence data')
    parser.add_argument('--output-dir', type=str, default='demo_output',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def demo_occlusion_handling():
    """Demonstrate occlusion handling capabilities"""
    print("\n🛡️ Occlusion Handling Demo")
    print("=" * 50)
    
    # Create sample overlapping detections
    detections = np.array([
        [100, 100, 200, 300],  # Person 1
        [150, 120, 250, 320],  # Person 2 (overlapping)
        [300, 150, 400, 350],  # Person 3 (separate)
        [160, 130, 260, 330],  # Person 4 (heavily overlapping with 1&2)
    ])
    
    print(f"📊 Analyzing {len(detections)} detections for overlaps...")
    
    # Initialize overlap analyzer
    overlap_analyzer = OverlapAnalyzer(overlap_threshold=0.3)
    
    # Compute overlap matrix
    overlap_matrix = overlap_analyzer.compute_overlap_matrix(detections)
    print(f"Overlap Matrix:\n{overlap_matrix}")
    
    # Analyze spatial relationships
    spatial_info = overlap_analyzer.analyze_spatial_relationships(detections)
    print(f"Distance Matrix:\n{spatial_info['distance_matrix']}")
    print(f"Size Ratio Matrix:\n{spatial_info['size_ratio_matrix']}")
    
    # Detect crowd situation
    is_crowd = detect_crowd_situations([{'to_tlwh': lambda: det} for det in detections])
    print(f"Crowd Situation Detected: {is_crowd}")
    
    # Compute crowd density
    density = compute_crowd_density(detections, (640, 480))
    print(f"Crowd Density: {density:.3f}")
    
    # Initialize occlusion state manager
    occlusion_manager = OcclusionStateManager()
    
    # Simulate track boxes for occlusion analysis
    track_boxes = {i: det for i, det in enumerate(detections)}
    occlusion_manager.update_occlusion_state(track_boxes, overlap_analyzer, frame_id=1)
    
    print(f"\nOcclusion Analysis Results:")
    for track_id in range(len(detections)):
        is_occluded = occlusion_manager.is_track_occluded(track_id)
        occlusion_level = occlusion_manager.get_occlusion_level(track_id)
        print(f"  Track {track_id}: Occluded={is_occluded}, Level={occlusion_level:.2%}")


def demo_enhanced_operations():
    """Demonstrate enhanced box operations"""
    print("\n🔧 Enhanced Operations Demo")
    print("=" * 50)
    
    # Create sample trajectory
    trajectory = [
        np.array([100, 100, 200, 300]),  # Frame 1
        np.array([105, 102, 205, 302]),  # Frame 2
        np.array([110, 105, 210, 305]),  # Frame 3
        np.array([140, 120, 240, 320]),  # Frame 4 (jump)
        np.array([145, 122, 245, 322]),  # Frame 5
    ]
    
    print(f"📊 Analyzing trajectory with {len(trajectory)} boxes...")
    
    # Compute motion vectors
    motion_vectors = []
    for i in range(1, len(trajectory)):
        motion = compute_motion_vector(trajectory[i-1], trajectory[i])
        motion_vectors.append(motion)
        print(f"Frame {i-1} -> {i}: Motion = {motion}")
    
    # Compute trajectory stability
    stability = compute_box_stability(trajectory)
    print(f"Trajectory Stability Score: {stability:.3f}")
    
    # Detect anomalies
    anomalies = detect_box_anomalies(trajectory, motion_threshold=20.0)
    print(f"Anomaly Detection: {anomalies}")
    
    # Smooth trajectory
    smoothed = smooth_box_trajectory(trajectory, alpha=0.7)
    print(f"Original vs Smoothed (Frame 3):")
    print(f"  Original: {trajectory[3]}")
    print(f"  Smoothed: {smoothed[3]}")
    
    # Compute overlaps between consecutive frames
    overlaps = []
    for i in range(1, len(trajectory)):
        overlap = compute_box_overlap(trajectory[i-1], trajectory[i])
        overlaps.append(overlap)
    print(f"Frame-to-frame overlaps: {overlaps}")
    
    # Demonstrate missing box interpolation
    trajectory_with_missing = trajectory.copy()
    trajectory_with_missing[2] = None  # Simulate missing detection
    trajectory_with_missing[3] = None
    
    interpolated = interpolate_missing_boxes(trajectory_with_missing)
    print(f"Interpolation completed: {len(interpolated)} boxes recovered")


def demo_visualization():
    """Demonstrate enhanced visualization capabilities"""
    print("\n📊 Enhanced Visualization Demo")
    print("=" * 50)
    
    if not CV2_AVAILABLE:
        print("⚠️ OpenCV not available, skipping visualization demo")
        return
    
    # Create plotter
    plotter = EnhancedMetricsPlotter('demo_output/visualization')
    
    # Generate sample metrics data
    sample_metrics = {
        'Enhanced StrongSort': [85, 78, 92, 88, 95],
        'Basic StrongSort': [75, 70, 80, 78, 82],
        'ByteTrack': [80, 75, 85, 82, 88]
    }
    
    metric_labels = ['MOTA', 'MOTP', 'IDF1', 'HOTA', 'Precision']
    
    print("📈 Generating radar chart...")
    plotter.plot_radar_chart(
        sample_metrics, 
        metric_labels,
        title='Enhanced Tracking Performance Comparison'
    )
    
    # Generate sample occlusion data
    occlusion_data = {
        'frames': list(range(1, 101)),
        'occlusion_levels': np.random.beta(2, 5, 100) * 0.8,  # Realistic occlusion pattern
        'track_counts': np.random.poisson(8, 100),
        'occlusion_types': {
            'No Occlusion': 60,
            'Partial Occlusion': 25,
            'Full Occlusion': 10,
            'Mutual Occlusion': 5
        },
        'id_switches_per_frame': np.random.poisson(0.2, 10),
        'track_qualities': np.random.beta(3, 2, 50),
        'track_occlusions': np.random.beta(2, 8, 50)
    }
    
    print("🛡️ Generating occlusion analysis plots...")
    plotter.plot_occlusion_analysis(occlusion_data)
    
    # Generate sample trajectory data
    trajectory_data = {}
    for track_id in range(5):
        # Generate realistic trajectory
        start_x, start_y = np.random.randint(100, 500), np.random.randint(100, 300)
        trajectory = []
        positions = []
        occlusion_levels = []
        
        for frame in range(50):
            # Add some noise and trend
            x = start_x + frame * 2 + np.random.normal(0, 5)
            y = start_y + np.random.normal(0, 3)
            positions.append([x, y])
            
            # Simulate occlusion levels
            if 20 <= frame <= 30:  # Occlusion period
                occlusion_levels.append(0.3 + np.random.random() * 0.5)
            else:
                occlusion_levels.append(np.random.random() * 0.2)
        
        trajectory_data[track_id] = {
            'positions': positions,
            'occlusion_levels': occlusion_levels,
            'frames': list(range(50))
        }
    
    print("🎯 Generating trajectory visualization...")
    plotter.plot_track_trajectories(
        trajectory_data,
        image_size=(640, 480),
        show_occlusion=True
    )
    
    # Generate comparison data
    comparison_data = {
        'Enhanced StrongSort': {'MOTA': 0.85, 'IDF1': 0.92, 'HOTA': 0.88},
        'Basic StrongSort': {'MOTA': 0.75, 'IDF1': 0.80, 'HOTA': 0.78},
        'ByteTrack': {'MOTA': 0.80, 'IDF1': 0.85, 'HOTA': 0.82}
    }
    
    print("📊 Generating performance comparison...")
    plotter.plot_performance_comparison(
        comparison_data,
        ['MOTA', 'IDF1', 'HOTA']
    )
    
    print(f"✅ All visualizations saved to: demo_output/visualization/")


def demo_metrics_evaluation():
    """Demonstrate metrics and evaluation capabilities"""
    print("\n📊 Metrics & Evaluation Demo")
    print("=" * 50)
    
    # Initialize tracking evaluator
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    
    # Simulate tracking data for several frames
    for frame_id in range(1, 51):
        # Generate ground truth detections
        gt_detections = []
        for track_id in range(3):
            # Simulate track movement
            x = 100 + track_id * 150 + frame_id * 2
            y = 100 + np.random.normal(0, 5)
            w, h = 80 + np.random.normal(0, 5), 180 + np.random.normal(0, 10)
            
            # Simulate occlusion
            occluded = (20 <= frame_id <= 30) and track_id == 1
            
            gt_detections.append({
                'id': track_id,
                'bbox': [x, y, x + w, y + h],
                'occluded': occluded
            })
        
        # Generate predicted detections (with some errors)
        pred_detections = []
        for track_id in range(3):
            # Add some prediction noise
            x = 100 + track_id * 150 + frame_id * 2 + np.random.normal(0, 3)
            y = 100 + np.random.normal(0, 8)
            w, h = 80 + np.random.normal(0, 8), 180 + np.random.normal(0, 15)
            
            # Simulate occasional missed detections
            if np.random.random() > 0.05:  # 95% detection rate
                pred_detections.append({
                    'id': track_id,
                    'bbox': [x, y, x + w, y + h],
                    'conf': 0.7 + np.random.random() * 0.3
                })
        
        # Add to evaluator
        processing_time = 0.02 + np.random.normal(0, 0.005)  # Simulate processing time
        evaluator.add_frame_data(frame_id, gt_detections, pred_detections, processing_time)
    
    # Compute metrics
    print("📊 Computing comprehensive metrics...")
    metrics = evaluator.compute_final_metrics()
    
    # Print summary
    print_metrics_summary(metrics, "Enhanced Tracking Evaluation Results")
    
    # Generate detailed report
    report = evaluator.get_detailed_report()
    print("\n📄 Detailed Metrics Report:")
    for category, values in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for metric, value in values.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Save report
    evaluator.save_report('demo_output/evaluation_report.json')
    
    # Demonstrate real-time monitoring
    print("\n⚡ Real-time Metrics Monitoring Demo:")
    monitor = RealTimeMetricsMonitor(window_size=20)
    
    for frame in range(20):
        # Simulate frame metrics
        frame_metrics = {
            'processing_time': 0.02 + np.random.normal(0, 0.005),
            'match_rate': 0.85 + np.random.normal(0, 0.1),
            'id_switches': np.random.poisson(0.1),
            'detection_count': np.random.poisson(8),
            'track_count': np.random.poisson(6),
            'new_tracks': np.random.poisson(0.5)
        }
        monitor.update(frame_metrics)
    
    current_metrics = monitor.get_current_metrics()
    print("Current Real-time Metrics:")
    for metric, value in current_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Check for quality alerts
    alerts = monitor.get_quality_alerts()
    print(f"Quality Alerts: {len(alerts)} recent alerts")


def demo_quality_assessment():
    """Demonstrate quality assessment and parameter optimization"""
    print("\n🎯 Quality Assessment & Parameter Optimization Demo")
    print("=" * 50)
    
    # Initialize tools
    analyzer = StrongSortAnalyzer()
    quality_assessor = QualityAssessor()
    parameter_tuner = ParameterTuner()
    
    # Simulate tracking data
    print("📊 Simulating tracking session...")
    
    for frame_id in range(100):
        # Simulate frame data
        matches = [(i, i) for i in range(np.random.poisson(5))]
        unmatched_tracks = list(range(len(matches), len(matches) + np.random.poisson(1)))
        unmatched_detections = list(range(len(matches), len(matches) + np.random.poisson(1)))
        processing_time = 0.025 + np.random.normal(0, 0.008)
        
        # Create mock tracker and detections
        class MockTracker:
            def __init__(self):
                self.tracks = [MockTrack(i) for i in range(len(matches) + len(unmatched_tracks))]
        
        class MockTrack:
            def __init__(self, track_id):
                self.id = track_id
                self.quality_score = 0.5 + np.random.random() * 0.5
                self.age = np.random.randint(1, 50)
                self.hit_streak = np.random.randint(1, 20)
        
        mock_tracker = MockTracker()
        detections = [{'confidence': 0.5 + np.random.random() * 0.5} for _ in range(len(matches) + len(unmatched_detections))]
        
        # Update analyzer
        analyzer.update_metrics(
            mock_tracker, detections, matches, 
            unmatched_tracks, unmatched_detections, 
            processing_time, frame_id
        )
        
        # Update quality assessor
        quality_assessor.assess_frame_quality(
            mock_tracker, detections, matches, frame_id
        )
    
    # Get analysis results
    print("📈 Generating performance analysis...")
    summary = analyzer.get_performance_summary()
    
    print("Performance Summary:")
    for category, metrics in summary.items():
        print(f"\n{category.title()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    
    # Get quality trends
    trends = quality_assessor.get_quality_trends()
    print(f"\nQuality Trends:")
    for metric, trend in trends.items():
        direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
        print(f"  {metric}: {direction} {trend:.3f}")
    
    # Get quality alerts
    alerts = quality_assessor.get_quality_alerts()
    print(f"\nQuality Alerts ({len(alerts)}):")
    for alert in alerts[:5]:  # Show first 5 alerts
        print(f"  • {alert}")
    
    # Parameter optimization
    print("\n⚙️ Parameter Optimization Recommendations:")
    recommendations = parameter_tuner.suggest_parameters(summary)
    for param, value in recommendations.items():
        print(f"  {param}: {value}")


def main():
    """Main demonstration function"""
    args = parse_args()
    
    print("🚀 Enhanced BoxMOT Utilities Demonstration")
    print("=" * 60)
    
    # Check utilities status
    print_utils_status()
    
    if not ENHANCED_UTILS_AVAILABLE:
        print("\n❌ Enhanced utilities not available. Please check installation.")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Show enhanced tracker configuration
    print(f"\n⚙️ Enhanced Tracker Configuration:")
    config = get_enhanced_tracker_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run selected demos
    if args.demo_type in ['all', 'occlusion']:
        demo_occlusion_handling()
    
    if args.demo_type in ['all', 'operations']:
        demo_enhanced_operations()
    
    if args.demo_type in ['all', 'visualization']:
        demo_visualization()
    
    if args.demo_type in ['all', 'metrics']:
        demo_metrics_evaluation()
        demo_quality_assessment()
    
    print(f"\n✅ Demo completed! Results saved to: {args.output_dir}")
    print("🎉 Enhanced BoxMOT utilities provide state-of-the-art tracking capabilities!")


if __name__ == '__main__':
    main() 