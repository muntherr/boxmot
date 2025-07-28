#!/usr/bin/env python3
"""
Enhanced StrongSort Demonstration Script

This script demonstrates the enhanced StrongSort tracker capabilities including:
- Improved ID preservation
- Enhanced ReID performance
- Advanced analytics and monitoring
- Parameter optimization
- Quality assessment

Author: Enhanced StrongSort Team
"""

import argparse
import time
from pathlib import Path
import sys
import numpy as np

# Add boxmot to path
sys.path.append(str(Path(__file__).parent.parent))

from boxmot import StrongSort
from boxmot.utils.strongsort_utils import (
    StrongSortAnalyzer, 
    ParameterTuner, 
    QualityAssessor,
    visualize_tracking_state,
    create_tracking_report
)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Visualization features will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GPU acceleration disabled.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced StrongSort Demonstration')
    
    parser.add_argument('--source', type=str, default='assets/MOT17-mini/train/MOT17-02-FRCNN',
                       help='Path to tracking sequence')
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt',
                       help='Path to ReID model weights')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu, cuda)')
    parser.add_argument('--conf-threshold', type=float, default=0.1,
                       help='Detection confidence threshold')
    parser.add_argument('--save-results', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--analyze', action='store_true',
                       help='Enable detailed analysis')
    parser.add_argument('--optimize-params', action='store_true',
                       help='Enable parameter optimization')
    
    return parser.parse_args()


def load_mot_sequence(sequence_path):
    """Load MOT sequence data"""
    sequence_path = Path(sequence_path)
    
    # Load detections
    det_file = sequence_path / 'det' / 'det.txt'
    if not det_file.exists():
        raise FileNotFoundError(f"Detection file not found: {det_file}")
    
    detections = {}
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                
                if frame_id not in detections:
                    detections[frame_id] = []
                
                # Convert to xyxy format and add class (0 for person)
                detections[frame_id].append([x, y, x+w, y+h, conf, 0])
    
    # Load images
    img_dir = sequence_path / 'img1'
    image_files = sorted(list(img_dir.glob('*.jpg')))
    
    return detections, image_files


def create_enhanced_tracker(args):
    """Create enhanced StrongSort tracker with optimized parameters"""
    
    # Enhanced parameters for better ID preservation
    enhanced_params = {
        'reid_weights': Path(args.reid_weights),
        'device': torch.device(args.device) if TORCH_AVAILABLE else 'cpu',
        'half': False,  # Disable half precision for better accuracy
        'per_class': False,
        'min_conf': args.conf_threshold,
        'max_cos_dist': 0.15,  # Stricter appearance matching
        'max_iou_dist': 0.7,
        'max_age': 50,  # Keep tracks longer
        'n_init': 2,  # Faster confirmation
        'nn_budget': 150,  # More ReID memory
        'mc_lambda': 0.995,  # Better motion consistency
        'ema_alpha': 0.9,
        # Enhanced parameters
        'conf_thresh_high': 0.7,
        'conf_thresh_low': 0.3,
        'id_preservation_weight': 0.1,
        'adaptive_matching': True,
        'appearance_weight': 0.6,
        'motion_weight': 0.4,
    }
    
    return StrongSort(**enhanced_params)


def simulate_detections_with_noise(detections, noise_level=0.1):
    """Add realistic noise to detections to test robustness"""
    noisy_detections = []
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        
        # Add position noise
        w, h = x2 - x1, y2 - y1
        noise_x = np.random.normal(0, w * noise_level)
        noise_y = np.random.normal(0, h * noise_level)
        
        # Add confidence noise
        conf_noise = np.random.normal(0, 0.05)
        new_conf = np.clip(conf + conf_noise, 0.1, 1.0)
        
        noisy_det = [
            x1 + noise_x, y1 + noise_y, 
            x2 + noise_x, y2 + noise_y, 
            new_conf, cls
        ]
        noisy_detections.append(noisy_det)
    
    return noisy_detections


def run_enhanced_tracking_demo(args):
    """Run the enhanced tracking demonstration"""
    
    print("🚀 Enhanced StrongSort Demonstration")
    print("=" * 50)
    
    # Setup
    results_dir = Path(args.save_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequence
    print("📁 Loading MOT sequence...")
    try:
        detections, image_files = load_mot_sequence(args.source)
        print(f"✅ Loaded {len(detections)} frames with detections")
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        return
    
    # Create enhanced tracker
    print("🔧 Creating enhanced StrongSort tracker...")
    try:
        tracker = create_enhanced_tracker(args)
        print("✅ Enhanced tracker created successfully")
    except Exception as e:
        print(f"❌ Error creating tracker: {e}")
        return
    
    # Initialize analysis tools
    analyzer = StrongSortAnalyzer() if args.analyze else None
    quality_assessor = QualityAssessor() if args.analyze else None
    parameter_tuner = ParameterTuner() if args.optimize_params else None
    
    print("\n🏃 Starting tracking...")
    print("-" * 30)
    
    # Tracking loop
    total_frames = len(image_files)
    total_processing_time = 0
    frame_results = []
    
    for frame_idx, img_file in enumerate(image_files):
        frame_id = frame_idx + 1
        
        # Load image
        if CV2_AVAILABLE:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"⚠️  Could not load image: {img_file}")
                continue
        else:
            # Create dummy image for non-CV2 environments
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get detections for this frame
        frame_detections = detections.get(frame_id, [])
        
        # Add some noise to test robustness
        if frame_detections:
            frame_detections = simulate_detections_with_noise(frame_detections, noise_level=0.05)
        
        # Convert to numpy array
        dets = np.array(frame_detections) if frame_detections else np.empty((0, 6))
        
        # Track
        start_time = time.time()
        try:
            tracks = tracker.update(dets, img)
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Analysis
            if analyzer:
                # Get matching info (simplified)
                matches = [(i, i) for i in range(min(len(dets), len(tracks)))]
                unmatched_tracks = list(range(len(matches), len(tracks)))
                unmatched_detections = list(range(len(matches), len(dets)))
                
                analyzer.update_metrics(
                    tracker.tracker, frame_detections, matches, 
                    unmatched_tracks, unmatched_detections, 
                    processing_time, frame_id
                )
            
            if quality_assessor:
                quality_scores = quality_assessor.assess_frame_quality(
                    tracker.tracker, frame_detections, matches, frame_id
                )
            
            # Store results
            frame_results.append({
                'frame_id': frame_id,
                'detections': len(frame_detections),
                'tracks': len(tracks),
                'processing_time_ms': processing_time * 1000,
            })
            
            # Visualization
            if args.visualize and CV2_AVAILABLE:
                vis_img = visualize_tracking_state(img, tracker.tracker, frame_detections)
                vis_path = results_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(vis_path), vis_img)
            
            # Progress update
            if frame_idx % 10 == 0:
                avg_time = total_processing_time / (frame_idx + 1) * 1000
                print(f"Frame {frame_id:4d}/{total_frames} | "
                      f"Tracks: {len(tracks):2d} | "
                      f"Dets: {len(frame_detections):2d} | "
                      f"Time: {processing_time*1000:5.1f}ms | "
                      f"Avg: {avg_time:5.1f}ms")
        
        except Exception as e:
            print(f"❌ Error processing frame {frame_id}: {e}")
            continue
    
    # Final statistics
    print("\n📊 Tracking Results Summary")
    print("=" * 40)
    
    if frame_results:
        avg_processing_time = np.mean([r['processing_time_ms'] for r in frame_results])
        avg_tracks = np.mean([r['tracks'] for r in frame_results])
        avg_detections = np.mean([r['detections'] for r in frame_results])
        fps = 1000 / avg_processing_time if avg_processing_time > 0 else 0
        
        print(f"📈 Frames Processed: {len(frame_results)}")
        print(f"⏱️  Average Processing Time: {avg_processing_time:.2f} ms")
        print(f"🎯 Average FPS: {fps:.1f}")
        print(f"👥 Average Tracks: {avg_tracks:.1f}")
        print(f"🔍 Average Detections: {avg_detections:.1f}")
        print(f"🆔 Next Track ID: {tracker.tracker._next_id}")
    
    # Advanced Analysis
    if analyzer:
        print("\n🔬 Performance Analysis")
        print("-" * 25)
        
        summary = analyzer.get_performance_summary()
        
        print(f"Match Rate: {summary.get('overall', {}).get('avg_match_rate', 0):.1%}")
        print(f"Track Quality: {summary.get('overall', {}).get('avg_track_quality', 0):.2f}")
        print(f"ID Switches: {summary.get('overall', {}).get('total_id_switches', 0)}")
        
        # Generate detailed report
        print("\n📄 Generating detailed performance report...")
        create_tracking_report(analyzer, str(results_dir / "analysis"))
        
        # Parameter optimization
        if parameter_tuner:
            print("\n⚙️  Parameter Optimization")
            print("-" * 27)
            
            recommendations = parameter_tuner.suggest_parameters(summary)
            print("Recommended parameters:")
            for param, value in recommendations.items():
                print(f"  {param}: {value}")
    
    # Quality Assessment
    if quality_assessor:
        print("\n🎯 Quality Assessment")
        print("-" * 20)
        
        trends = quality_assessor.get_quality_trends()
        alerts = quality_assessor.get_quality_alerts()
        
        if trends:
            print("Quality trends:")
            for metric, trend in trends.items():
                direction = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
                print(f"  {metric}: {direction} {trend:.3f}")
        
        if alerts:
            print("\n⚠️  Quality Alerts:")
            for alert in alerts:
                print(f"  • {alert}")
    
    # Tracker statistics
    if hasattr(tracker, 'get_track_statistics'):
        print("\n📊 Tracker Statistics")
        print("-" * 21)
        
        stats = tracker.get_track_statistics()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Results saved to: {results_dir}")
    print("🎉 Enhanced StrongSort demonstration completed!")


def main():
    """Main function"""
    args = parse_args()
    
    try:
        run_enhanced_tracking_demo(args)
    except KeyboardInterrupt:
        print("\n⏹️  Tracking interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 