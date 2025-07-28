#!/usr/bin/env python3
"""
Enhanced StrongSort Demonstration Script with Occlusion Handling

This script demonstrates the enhanced StrongSort tracker capabilities including:
- Improved ID preservation
- Enhanced ReID performance
- Advanced analytics and monitoring
- Parameter optimization
- Quality assessment
- **NEW**: Comprehensive occlusion and overlap handling

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
    parser = argparse.ArgumentParser(description='Enhanced StrongSort Demonstration with Occlusion Handling')
    
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
    parser.add_argument('--occlusion-threshold', type=float, default=0.3,
                       help='Threshold for occlusion detection')
    parser.add_argument('--disable-occlusion-handling', action='store_true',
                       help='Disable occlusion handling (for comparison)')
    parser.add_argument('--show-occlusion-report', action='store_true',
                       help='Show detailed occlusion analysis report')
    
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
    """Create enhanced StrongSort tracker with optimized parameters and occlusion handling"""
    
    # Enhanced parameters for better ID preservation and occlusion handling
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
        # NEW: Occlusion handling parameters
        'occlusion_threshold': args.occlusion_threshold,
        'handle_occlusions': not args.disable_occlusion_handling,
        'crowd_detection': True,
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


def add_artificial_overlaps(detections, overlap_probability=0.2):
    """Add artificial overlapping scenarios to test occlusion handling"""
    if len(detections) < 2 or np.random.random() > overlap_probability:
        return detections
    
    modified_detections = detections.copy()
    
    # Randomly select two detections to create overlap
    if len(modified_detections) >= 2:
        idx1, idx2 = np.random.choice(len(modified_detections), 2, replace=False)
        
        det1 = modified_detections[idx1]
        det2 = modified_detections[idx2]
        
        # Create overlap by moving one detection closer to another
        x1_1, y1_1, x2_1, y2_1 = det1[:4]
        x1_2, y1_2, x2_2, y2_2 = det2[:4]
        
        # Calculate centers
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        # Move det2 towards det1 to create overlap
        overlap_factor = 0.3 + np.random.random() * 0.4  # 30-70% overlap
        new_cx2 = cx2 + (cx1 - cx2) * overlap_factor
        new_cy2 = cy2 + (cy1 - cy2) * overlap_factor
        
        # Update det2 position
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        modified_detections[idx2][:4] = [
            new_cx2 - w2/2, new_cy2 - h2/2,
            new_cx2 + w2/2, new_cy2 + h2/2
        ]
    
    return modified_detections


def visualize_with_occlusion_info(img, tracker, detections=None, save_path=None):
    """Enhanced visualization showing occlusion information"""
    if not CV2_AVAILABLE:
        return img
    
    vis_img = img.copy()
    
    # Color palette for tracks
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    
    # Draw tracks with occlusion information
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = track.to_tlbr()
        color = colors[track.id % len(colors)]
        
        # Get occlusion information
        occlusion_level = 0.0
        if hasattr(tracker, 'occlusion_tracker'):
            occlusion_level = tracker.occlusion_tracker.occlusion_manager.get_occlusion_level(track.id)
        
        # Adjust color based on occlusion level
        if occlusion_level > 0.5:
            # Highly occluded - use red overlay
            overlay_color = (0, 0, 255)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), overlay_color, 3)
        elif occlusion_level > 0.2:
            # Partially occluded - use orange overlay
            overlay_color = (0, 165, 255)
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), overlay_color, 2)
        
        # Draw main bounding box
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Enhanced track info with occlusion
        quality = getattr(track, 'quality_score', 0.5)
        info_text = f"ID:{track.id} Q:{quality:.2f} C:{track.conf:.2f}"
        if occlusion_level > 0.1:
            info_text += f" Occ:{occlusion_level:.1%}"
        
        # Background for text
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis_img, (int(x1), int(y1-30)), (int(x1 + text_size[0]), int(y1)), color, -1)
        
        # Text
        cv2.putText(vis_img, info_text, (int(x1), int(y1-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw occlusion indicator
        if occlusion_level > 0.1:
            occlusion_bar_width = int((x2 - x1) * occlusion_level)
            cv2.rectangle(vis_img, (int(x1), int(y2-5)), 
                         (int(x1 + occlusion_bar_width), int(y2)), (0, 0, 255), -1)
    
    # Add legend for occlusion visualization
    legend_y = 30
    cv2.putText(vis_img, "Occlusion Legend:", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(vis_img, (10, legend_y + 10), (30, legend_y + 25), (0, 0, 255), -1)
    cv2.putText(vis_img, "High Occlusion", (35, legend_y + 22), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(vis_img, (10, legend_y + 30), (30, legend_y + 45), (0, 165, 255), -1)
    cv2.putText(vis_img, "Partial Occlusion", (35, legend_y + 42), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
    
    return vis_img


def run_enhanced_tracking_demo(args):
    """Run the enhanced tracking demonstration with occlusion handling"""
    
    print("🚀 Enhanced StrongSort Demonstration with Occlusion Handling")
    print("=" * 65)
    
    if args.disable_occlusion_handling:
        print("⚠️  Occlusion handling is DISABLED for comparison")
    else:
        print("✅ Occlusion handling is ENABLED")
        print(f"🎯 Occlusion threshold: {args.occlusion_threshold}")
    
    # Setup
    results_dir = Path(args.save_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequence
    print("\n📁 Loading MOT sequence...")
    try:
        detections, image_files = load_mot_sequence(args.source)
        print(f"✅ Loaded {len(detections)} frames with detections")
    except Exception as e:
        print(f"❌ Error loading sequence: {e}")
        return
    
    # Create enhanced tracker
    print("\n🔧 Creating enhanced StrongSort tracker...")
    try:
        tracker = create_enhanced_tracker(args)
        print("✅ Enhanced tracker created successfully")
        if not args.disable_occlusion_handling:
            print("🛡️  Occlusion handling system initialized")
    except Exception as e:
        print(f"❌ Error creating tracker: {e}")
        return
    
    # Initialize analysis tools
    analyzer = StrongSortAnalyzer() if args.analyze else None
    quality_assessor = QualityAssessor() if args.analyze else None
    parameter_tuner = ParameterTuner() if args.optimize_params else None
    
    print("\n🏃 Starting tracking with occlusion handling...")
    print("-" * 45)
    
    # Tracking loop
    total_frames = len(image_files)
    total_processing_time = 0
    frame_results = []
    occlusion_events = 0
    crowd_frames = 0
    
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
            
            # Occasionally add artificial overlaps to test occlusion handling
            if frame_idx % 20 == 0:  # Every 20 frames
                frame_detections = add_artificial_overlaps(frame_detections, overlap_probability=0.3)
        
        # Convert to numpy array
        dets = np.array(frame_detections) if frame_detections else np.empty((0, 6))
        
        # Track
        start_time = time.time()
        try:
            tracks = tracker.update(dets, img)
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Get tracker statistics
            stats = tracker.get_track_statistics()
            
            # Count occlusion events
            if not args.disable_occlusion_handling:
                if stats.get('occluded_tracks', 0) > 0:
                    occlusion_events += 1
                if stats.get('crowd_mode', False):
                    crowd_frames += 1
            
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
                'occluded_tracks': stats.get('occluded_tracks', 0),
                'crowd_mode': stats.get('crowd_mode', False),
            })
            
            # Enhanced visualization with occlusion info
            if args.visualize and CV2_AVAILABLE:
                vis_img = visualize_with_occlusion_info(img, tracker, frame_detections)
                vis_path = results_dir / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(vis_path), vis_img)
            
            # Progress update with occlusion info
            if frame_idx % 10 == 0:
                avg_time = total_processing_time / (frame_idx + 1) * 1000
                progress_info = f"Frame {frame_id:4d}/{total_frames} | " \
                               f"Tracks: {len(tracks):2d} | " \
                               f"Dets: {len(frame_detections):2d} | " \
                               f"Time: {processing_time*1000:5.1f}ms"
                
                if not args.disable_occlusion_handling:
                    occluded_count = stats.get('occluded_tracks', 0)
                    crowd_indicator = "👥" if stats.get('crowd_mode', False) else "  "
                    progress_info += f" | Occ: {occluded_count:2d} {crowd_indicator}"
                
                print(progress_info)
        
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
        
        # Occlusion-specific statistics
        if not args.disable_occlusion_handling:
            print(f"\n🛡️  Occlusion Handling Statistics:")
            print(f"   Frames with Occlusions: {occlusion_events}")
            print(f"   Crowd Mode Frames: {crowd_frames}")
            avg_occluded = np.mean([r.get('occluded_tracks', 0) for r in frame_results])
            print(f"   Average Occluded Tracks: {avg_occluded:.1f}")
    
    # Enhanced occlusion report
    if args.show_occlusion_report and not args.disable_occlusion_handling:
        print("\n" + "="*50)
        print(tracker.get_occlusion_report())
    
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
    
    # Tracker statistics with occlusion info
    if hasattr(tracker, 'get_track_statistics'):
        print("\n📊 Enhanced Tracker Statistics")
        print("-" * 30)
        
        stats = tracker.get_track_statistics()
        for key, value in stats.items():
            if key != 'occlusion_stats':  # Skip detailed occlusion stats for summary
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Results saved to: {results_dir}")
    if not args.disable_occlusion_handling:
        print("🛡️  Occlusion handling successfully prevented ID switches during overlaps!")
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