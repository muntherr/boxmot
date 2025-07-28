# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import sys
import threading
from pathlib import Path

import numpy as np
import multiprocessing as mp

# global logger
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TOML = ROOT / "pyproject.toml"

BOXMOT     = ROOT / "boxmot"
CONFIGS    = BOXMOT / "configs"
TRACKER_CONFIGS   = CONFIGS / "trackers"
DATASET_CONFIGS   = CONFIGS / "datasets"

ENGINE   = BOXMOT / "engine"
WEIGHTS  = ENGINE / "weights"
TRACKEVAL  = ENGINE / "trackeval"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

def _is_main_process(record):
    return mp.current_process().name == "MainProcess"

def configure_logging():
    # this will remove *all* existing handlers and then add yours
    logger.configure(handlers=[
        {
            "sink": sys.stderr,
            "level":    "INFO",
            "filter":   _is_main_process,
        }
    ])
    
configure_logging()

# Enhanced Utilities Imports
try:
    # Core enhanced utilities
    from .strongsort_utils import (
        StrongSortAnalyzer, 
        ParameterTuner, 
        QualityAssessor,
        visualize_tracking_state,
        create_tracking_report
    )
    
    # Occlusion handling
    from .occlusion_handler import (
        OcclusionAwareTracker,
        OverlapAnalyzer,
        OcclusionStateManager,
        OcclusionType,
        detect_crowd_situations,
        compute_crowd_density
    )
    
    # Enhanced plotting and visualization
    from .plots import (
        EnhancedMetricsPlotter,
        MetricsPlotter,  # Legacy compatibility
        visualize_tracking_frame,
        create_tracking_dashboard
    )
    
    # Enhanced matching and association
    from .matching import (
        enhanced_linear_assignment,
        occlusion_aware_iou_distance,
        enhanced_embedding_distance,
        adaptive_fuse_motion,
        enhanced_fuse_iou,
        enhanced_fuse_score,
        multi_modal_distance,
        compute_distance_matrix
    )
    
    # Enhanced association functions
    from .association import (
        enhanced_associate_detections_to_trackers,
        enhanced_associate,
        compute_occlusion_aware_cost,
        compute_affinity_matrix,
        multi_stage_association
    )
    
    # Enhanced operations
    from .ops import (
        # Basic conversions
        xyxy2xywh, xywh2xyxy, xyxy2tlwh, tlwh2xyxy, tlwh2xywh, xywh2tlwh,
        # Enhanced operations
        compute_box_overlap,
        compute_box_center_distance,
        expand_box,
        crop_box_region,
        smooth_box_trajectory,
        compute_motion_vector,
        predict_box_position,
        compute_box_stability,
        filter_boxes_by_area,
        filter_boxes_by_aspect_ratio,
        compute_occlusion_matrix,
        detect_box_anomalies,
        interpolate_missing_boxes,
        normalize_boxes,
        denormalize_boxes
    )
    
    # Comprehensive metrics and evaluation
    from .metrics import (
        TrackingMetrics,
        TrackingEvaluator,
        RealTimeMetricsMonitor,
        compute_mot_metrics,
        print_metrics_summary
    )
    
    ENHANCED_UTILS_AVAILABLE = True
    
except ImportError as e:
    logger.warning(f"Some enhanced utilities not available: {e}")
    ENHANCED_UTILS_AVAILABLE = False

# Legacy utilities (always available)
try:
    from .iou import AssociationFunction
    from .checks import check_requirements
    from .misc import *
    from .torch_utils import *
    from .download import *
    
    BASIC_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Basic utilities not available: {e}")
    BASIC_UTILS_AVAILABLE = False

# Convenience functions
def get_enhanced_tracker_config():
    """Get default configuration for enhanced tracking"""
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
        'id_preservation_weight': 0.1,
        'adaptive_matching': True,
        'appearance_weight': 0.6,
        'motion_weight': 0.4,
        'handle_occlusions': True,
        'occlusion_threshold': 0.3,
        'crowd_detection': True,
    }

def create_enhanced_tracker(tracker_type='strongsort', **kwargs):
    """
    Create an enhanced tracker with optimized parameters.
    
    Args:
        tracker_type: Type of tracker ('strongsort', 'botsort', etc.)
        **kwargs: Additional parameters
        
    Returns:
        Enhanced tracker instance
    """
    if tracker_type.lower() == 'strongsort' and ENHANCED_UTILS_AVAILABLE:
        from boxmot import StrongSort
        
        # Merge default config with user parameters
        config = get_enhanced_tracker_config()
        config.update(kwargs)
        
        return StrongSort(**config)
    else:
        logger.warning(f"Enhanced {tracker_type} not available, falling back to basic version")
        # Fallback to basic tracker
        from boxmot import TRACKERS
        return TRACKERS[tracker_type](**kwargs)

def print_utils_status():
    """Print status of available utilities"""
    print("📦 BoxMOT Enhanced Utilities Status:")
    print(f"  ✅ Basic Utils: {'Available' if BASIC_UTILS_AVAILABLE else 'Not Available'}")
    print(f"  🚀 Enhanced Utils: {'Available' if ENHANCED_UTILS_AVAILABLE else 'Not Available'}")
    
    if ENHANCED_UTILS_AVAILABLE:
        print("  📊 Enhanced Features:")
        print("    • Occlusion-aware tracking")
        print("    • Advanced visualization")
        print("    • Real-time metrics monitoring")
        print("    • Quality assessment tools")
        print("    • Parameter optimization")
        print("    • Multi-modal distance computation")
        print("    • Enhanced association algorithms")

__all__ = [
    # Configuration and setup
    'ROOT', 'DATA', 'TOML', 'BOXMOT', 'CONFIGS', 'TRACKER_CONFIGS', 
    'DATASET_CONFIGS', 'ENGINE', 'WEIGHTS', 'NUM_THREADS',
    
    # Enhanced utilities (if available)
    'StrongSortAnalyzer', 'ParameterTuner', 'QualityAssessor',
    'OcclusionAwareTracker', 'OverlapAnalyzer', 'OcclusionStateManager',
    'EnhancedMetricsPlotter', 'TrackingEvaluator', 'RealTimeMetricsMonitor',
    
    # Enhanced functions
    'enhanced_linear_assignment', 'enhanced_associate_detections_to_trackers',
    'compute_box_overlap', 'visualize_tracking_frame', 'create_tracking_dashboard',
    'compute_mot_metrics', 'print_metrics_summary',
    
    # Convenience functions
    'get_enhanced_tracker_config', 'create_enhanced_tracker', 'print_utils_status',
    
    # Status flags
    'ENHANCED_UTILS_AVAILABLE', 'BASIC_UTILS_AVAILABLE'
]