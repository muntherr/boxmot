# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import lap
import numpy as np
import scipy
import torch
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Optional, Union

from boxmot.utils.iou import AssociationFunction

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def enhanced_linear_assignment(cost_matrix, thresh=None, method='lap'):
    """
    Enhanced linear assignment with multiple solver options and better error handling.
    
    Args:
        cost_matrix (np.ndarray): Cost matrix for assignment
        thresh (float, optional): Cost threshold for valid assignments
        method (str): Assignment method ('lap', 'scipy', 'greedy')
        
    Returns:
        tuple: (matches, unmatched_a, unmatched_b)
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    
    matches, unmatched_a, unmatched_b = [], [], []
    
    try:
        if method == 'lap':
            # LAP solver (fastest)
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, 
                                  cost_limit=thresh if thresh is not None else None)
            for ix, mx in enumerate(x):
                if mx >= 0:
                    if thresh is None or cost_matrix[ix, mx] <= thresh:
                        matches.append([ix, mx])
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
            
        elif method == 'scipy':
            # SciPy solver (more stable)
            from scipy.optimize import linear_sum_assignment
            
            if thresh is not None:
                cost_matrix = cost_matrix.copy()
                cost_matrix[cost_matrix > thresh] = 1e6
            
            x, y = linear_sum_assignment(cost_matrix)
            for i, j in zip(x, y):
                if thresh is None or cost_matrix[i, j] <= thresh:
                    matches.append([i, j])
            
            matched_rows = set([m[0] for m in matches])
            matched_cols = set([m[1] for m in matches])
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
            unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
            
        else:  # Greedy assignment
            cost_matrix_copy = cost_matrix.copy()
            while True:
                min_cost = np.min(cost_matrix_copy)
                if thresh is not None and min_cost > thresh:
                    break
                    
                min_idx = np.unravel_index(np.argmin(cost_matrix_copy), cost_matrix_copy.shape)
                matches.append([min_idx[0], min_idx[1]])
                
                # Remove assigned row and column
                cost_matrix_copy[min_idx[0], :] = np.inf
                cost_matrix_copy[:, min_idx[1]] = np.inf
                
                if np.all(np.isinf(cost_matrix_copy)):
                    break
            
            matched_rows = set([m[0] for m in matches])
            matched_cols = set([m[1] for m in matches])
            unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
            unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
    
    except Exception as e:
        # Fallback to simple greedy assignment
        matches, unmatched_a, unmatched_b = _greedy_assignment(cost_matrix, thresh)
    
    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_a), np.array(unmatched_b)


def _greedy_assignment(cost_matrix, thresh=None):
    """Fallback greedy assignment algorithm"""
    matches = []
    cost_matrix_copy = cost_matrix.copy()
    
    while True:
        min_cost = np.min(cost_matrix_copy)
        if thresh is not None and min_cost > thresh:
            break
            
        min_idx = np.unravel_index(np.argmin(cost_matrix_copy), cost_matrix_copy.shape)
        matches.append([min_idx[0], min_idx[1]])
        
        cost_matrix_copy[min_idx[0], :] = np.inf
        cost_matrix_copy[:, min_idx[1]] = np.inf
        
        if np.all(np.isinf(cost_matrix_copy)):
            break
    
    matched_rows = set([m[0] for m in matches])
    matched_cols = set([m[1] for m in matches])
    unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
    unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
    
    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """Legacy function for backward compatibility"""
    matches, unmatched_a, unmatched_b = enhanced_linear_assignment(cost_matrix, thresh)
    return matches, unmatched_a, unmatched_b


def occlusion_aware_iou_distance(atracks, btracks, 
                                atracks_occlusion=None, 
                                btracks_occlusion=None,
                                occlusion_penalty=0.1):
    """
    Compute IoU distance with occlusion awareness.
    
    Args:
        atracks: First set of tracks/detections
        btracks: Second set of tracks/detections  
        atracks_occlusion: Occlusion levels for first set
        btracks_occlusion: Occlusion levels for second set
        occlusion_penalty: Penalty weight for occluded objects
        
    Returns:
        Enhanced IoU distance matrix
    """
    # Compute base IoU distance
    base_distance = iou_distance(atracks, btracks)
    
    if atracks_occlusion is None and btracks_occlusion is None:
        return base_distance
    
    # Apply occlusion penalties
    distance_matrix = base_distance.copy()
    
    if atracks_occlusion is not None:
        for i, occlusion_level in enumerate(atracks_occlusion):
            if occlusion_level > 0.3:  # Partially or fully occluded
                # Increase distance for occluded tracks
                distance_matrix[i, :] *= (1 + occlusion_penalty * occlusion_level)
    
    if btracks_occlusion is not None:
        for j, occlusion_level in enumerate(btracks_occlusion):
            if occlusion_level > 0.3:
                distance_matrix[:, j] *= (1 + occlusion_penalty * occlusion_level)
    
    return distance_matrix


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU with improved handling of different input types.
    
    Args:
        atracks: List of tracks or numpy array of bounding boxes
        btracks: List of tracks or numpy array of bounding boxes
        
    Returns:
        Cost matrix based on 1 - IoU
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # Handle track objects
        atlbrs = []
        for track in atracks:
            if hasattr(track, 'xyxy'):
                atlbrs.append(track.xyxy)
            elif hasattr(track, 'to_tlbr'):
                atlbrs.append(track.to_tlbr())
            else:
                atlbrs.append(track)
        
        btlbrs = []
        for track in btracks:
            if hasattr(track, 'xyxy'):
                btlbrs.append(track.xyxy)
            elif hasattr(track, 'to_tlbr'):
                btlbrs.append(track.to_tlbr())
            else:
                btlbrs.append(track)

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    _ious = AssociationFunction.iou_batch(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def enhanced_embedding_distance(tracks, detections, 
                              metric="cosine",
                              track_qualities=None,
                              detection_confidences=None,
                              occlusion_info=None,
                              adaptive_threshold=True):
    """
    Enhanced embedding distance with quality weighting and occlusion awareness.
    
    Args:
        tracks: List of track objects
        detections: List of detection objects
        metric: Distance metric ('cosine', 'euclidean')
        track_qualities: Quality scores for tracks
        detection_confidences: Confidence scores for detections
        occlusion_info: Dictionary with occlusion information
        adaptive_threshold: Whether to use adaptive thresholding
        
    Returns:
        Enhanced cost matrix
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix

    # Extract features
    det_features = []
    for det in detections:
        if hasattr(det, 'curr_feat') and det.curr_feat is not None:
            det_features.append(det.curr_feat)
        elif hasattr(det, 'feature') and det.feature is not None:
            det_features.append(det.feature)
        else:
            # Use dummy feature if not available
            det_features.append(np.zeros(512))
    
    track_features = []
    for track in tracks:
        if hasattr(track, 'smooth_feat') and track.smooth_feat is not None:
            track_features.append(track.smooth_feat)
        elif hasattr(track, 'features') and track.features:
            # Use latest feature
            track_features.append(track.features[-1])
        else:
            track_features.append(np.zeros(512))
    
    if not det_features or not track_features:
        return cost_matrix
    
    det_features = np.asarray(det_features, dtype=np.float32)
    track_features = np.asarray(track_features, dtype=np.float32)
    
    # Normalize features for better distance computation
    det_features = det_features / (np.linalg.norm(det_features, axis=1, keepdims=True) + 1e-8)
    track_features = track_features / (np.linalg.norm(track_features, axis=1, keepdims=True) + 1e-8)
    
    # Compute base distance matrix
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    
    # Apply quality weighting
    if track_qualities is not None:
        # Lower cost for high-quality tracks
        quality_weight = np.expand_dims(track_qualities, axis=1)
        cost_matrix *= (2.0 - quality_weight)
    
    if detection_confidences is not None:
        # Lower cost for high-confidence detections
        conf_weight = np.expand_dims(detection_confidences, axis=0)
        cost_matrix *= (2.0 - conf_weight)
    
    # Apply occlusion adjustments
    if occlusion_info:
        track_occlusion = occlusion_info.get('track_occlusion_levels')
        det_occlusion = occlusion_info.get('detection_occlusion_levels')
        
        if track_occlusion is not None:
            for i, occlusion_level in enumerate(track_occlusion):
                if occlusion_level > 0.5:  # Highly occluded track
                    # Reduce appearance matching confidence
                    cost_matrix[i, :] *= (1 + 0.3 * occlusion_level)
        
        if det_occlusion is not None:
            for j, occlusion_level in enumerate(det_occlusion):
                if occlusion_level > 0.5:  # Highly occluded detection
                    cost_matrix[:, j] *= (1 + 0.3 * occlusion_level)
    
    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    """Legacy embedding distance function for backward compatibility"""
    return enhanced_embedding_distance(tracks, detections, metric)


def adaptive_fuse_motion(kf, cost_matrix, tracks, detections, 
                        only_position=False, lambda_=0.98,
                        track_qualities=None, adaptive_gating=True):
    """
    Enhanced motion fusion with adaptive parameters and quality weighting.
    
    Args:
        kf: Kalman filter object
        cost_matrix: Base cost matrix
        tracks: List of track objects
        detections: List of detection objects
        only_position: Whether to use only position for gating
        lambda_: Fusion weight for motion
        track_qualities: Quality scores for tracks
        adaptive_gating: Whether to use adaptive gating thresholds
        
    Returns:
        Enhanced cost matrix with motion fusion
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    gating_dim = 2 if only_position else 4
    base_threshold = chi2inv95[gating_dim]
    
    # Extract measurements
    measurements = []
    for det in detections:
        if hasattr(det, 'to_xyah'):
            measurements.append(det.to_xyah())
        else:
            # Convert from different formats
            if len(det) >= 4:
                x, y, w, h = det[:4]
                cx, cy = x + w/2, y + h/2
                aspect = w / h if h > 0 else 1.0
                measurements.append([cx, cy, aspect, h])
            else:
                measurements.append([0, 0, 1, 1])  # Default measurement
    
    measurements = np.asarray(measurements)
    
    for row, track in enumerate(tracks):
        try:
            # Compute gating distance
            gating_distance = kf.gating_distance(
                track.mean, track.covariance, measurements, only_position, metric="maha"
            )
            
            # Adaptive gating threshold based on track quality
            if adaptive_gating and track_qualities is not None:
                quality = track_qualities[row]
                # Higher quality tracks get more lenient gating
                adaptive_threshold = base_threshold * (2.0 - quality)
            else:
                adaptive_threshold = base_threshold
            
            # Apply gating
            cost_matrix[row, gating_distance > adaptive_threshold] = np.inf
            
            # Adaptive lambda based on track confidence
            if track_qualities is not None:
                track_lambda = lambda_ * track_qualities[row]
            else:
                track_lambda = lambda_
            
            # Fuse motion information
            cost_matrix[row] = track_lambda * cost_matrix[row] + (1 - track_lambda) * gating_distance
            
        except Exception as e:
            # Fallback: just apply basic gating
            if hasattr(track, 'mean') and hasattr(track, 'covariance'):
                try:
                    gating_distance = kf.gating_distance(
                        track.mean, track.covariance, measurements, only_position, metric="maha"
                    )
                    cost_matrix[row, gating_distance > base_threshold] = np.inf
                except:
                    pass  # Skip motion fusion for this track
    
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """Legacy motion fusion function for backward compatibility"""
    return adaptive_fuse_motion(kf, cost_matrix, tracks, detections, only_position, lambda_)


def enhanced_fuse_iou(cost_matrix, tracks, detections, 
                     track_qualities=None, 
                     detection_confidences=None,
                     iou_weight=0.5, reid_weight=0.5):
    """
    Enhanced IoU fusion with quality weighting and adaptive weights.
    
    Args:
        cost_matrix: Base ReID cost matrix
        tracks: List of track objects
        detections: List of detection objects
        track_qualities: Quality scores for tracks
        detection_confidences: Confidence scores for detections  
        iou_weight: Weight for IoU component
        reid_weight: Weight for ReID component
        
    Returns:
        Fused cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # Compute IoU similarity
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    
    # Adaptive weighting based on quality
    if track_qualities is not None or detection_confidences is not None:
        # Adjust weights based on confidence/quality
        adaptive_iou_weight = np.full_like(cost_matrix, iou_weight)
        adaptive_reid_weight = np.full_like(cost_matrix, reid_weight)
        
        if track_qualities is not None:
            for i, quality in enumerate(track_qualities):
                # High-quality tracks rely more on appearance
                if quality > 0.7:
                    adaptive_reid_weight[i, :] = reid_weight * 1.2
                    adaptive_iou_weight[i, :] = iou_weight * 0.8
                elif quality < 0.3:
                    # Low-quality tracks rely more on motion/IoU
                    adaptive_iou_weight[i, :] = iou_weight * 1.2
                    adaptive_reid_weight[i, :] = reid_weight * 0.8
        
        if detection_confidences is not None:
            for j, conf in enumerate(detection_confidences):
                if conf > 0.7:
                    adaptive_reid_weight[:, j] *= 1.1
                elif conf < 0.3:
                    adaptive_iou_weight[:, j] *= 1.1
        
        # Normalize weights
        total_weight = adaptive_iou_weight + adaptive_reid_weight
        adaptive_iou_weight /= total_weight
        adaptive_reid_weight /= total_weight
        
        fuse_sim = adaptive_reid_weight * reid_sim + adaptive_iou_weight * iou_sim
    else:
        # Standard fusion
        fuse_sim = reid_weight * reid_sim + iou_weight * iou_sim
    
    # Apply detection confidence weighting
    if detection_confidences is not None:
        det_confs = np.expand_dims(detection_confidences, axis=0)
        det_confs = np.repeat(det_confs, cost_matrix.shape[0], axis=0)
        fuse_sim = fuse_sim * (1 + det_confs) / 2
    
    fuse_cost = 1 - fuse_sim
    return np.clip(fuse_cost, 0, 2)  # Clip to reasonable range


def fuse_iou(cost_matrix, tracks, detections):
    """Legacy IoU fusion function for backward compatibility"""
    return enhanced_fuse_iou(cost_matrix, tracks, detections)


def enhanced_fuse_score(cost_matrix, detections, 
                       confidence_threshold=0.5,
                       confidence_weighting=True):
    """
    Enhanced score fusion with adaptive confidence weighting.
    
    Args:
        cost_matrix: Base cost matrix
        detections: List of detection objects
        confidence_threshold: Minimum confidence threshold
        confidence_weighting: Whether to apply confidence weighting
        
    Returns:
        Score-weighted cost matrix
    """
    if cost_matrix.size == 0:
        return cost_matrix
    
    # Extract detection confidences
    det_confs = []
    for det in detections:
        if hasattr(det, 'conf'):
            det_confs.append(det.conf)
        elif hasattr(det, 'confidence'):
            det_confs.append(det.confidence)
        elif isinstance(det, (list, tuple, np.ndarray)) and len(det) > 4:
            det_confs.append(det[4])
        else:
            det_confs.append(1.0)  # Default confidence
    
    det_confs = np.array(det_confs)
    
    if confidence_weighting:
        # Apply confidence weighting
        iou_sim = 1 - cost_matrix
        det_confs_expanded = np.expand_dims(det_confs, axis=0)
        det_confs_expanded = np.repeat(det_confs_expanded, cost_matrix.shape[0], axis=0)
        
        # Enhanced fusion with confidence thresholding
        confidence_mask = det_confs_expanded >= confidence_threshold
        
        # Apply stronger weighting for high-confidence detections
        confidence_weight = np.where(
            det_confs_expanded > 0.7,
            det_confs_expanded * 1.2,  # Boost high confidence
            det_confs_expanded
        )
        
        fuse_sim = iou_sim * confidence_weight * confidence_mask
        fuse_cost = 1 - fuse_sim
        
        # Penalize low-confidence detections
        fuse_cost = np.where(
            det_confs_expanded < confidence_threshold,
            fuse_cost * 2.0,  # Increase cost for low confidence
            fuse_cost
        )
    else:
        # Simple confidence gating
        det_confs_expanded = np.expand_dims(det_confs, axis=0)
        det_confs_expanded = np.repeat(det_confs_expanded, cost_matrix.shape[0], axis=0)
        
        # Mask out low-confidence detections
        low_conf_mask = det_confs_expanded < confidence_threshold
        fuse_cost = cost_matrix.copy()
        fuse_cost[low_conf_mask] = np.inf
    
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """Legacy score fusion function for backward compatibility"""
    return enhanced_fuse_score(cost_matrix, detections)


def multi_modal_distance(tracks, detections,
                        appearance_weight=0.4,
                        motion_weight=0.3, 
                        iou_weight=0.3,
                        kf=None,
                        track_qualities=None,
                        detection_confidences=None,
                        occlusion_info=None):
    """
    Compute multi-modal distance combining appearance, motion, and geometric cues.
    
    Args:
        tracks: List of track objects
        detections: List of detection objects
        appearance_weight: Weight for appearance similarity
        motion_weight: Weight for motion consistency
        iou_weight: Weight for IoU overlap
        kf: Kalman filter for motion prediction
        track_qualities: Quality scores for tracks
        detection_confidences: Confidence scores for detections
        occlusion_info: Dictionary with occlusion information
        
    Returns:
        Combined distance matrix
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))
    
    # Initialize combined cost matrix
    combined_cost = np.zeros((len(tracks), len(detections)))
    total_weight = 0
    
    # Appearance component
    if appearance_weight > 0:
        appearance_cost = enhanced_embedding_distance(
            tracks, detections, track_qualities=track_qualities,
            detection_confidences=detection_confidences, occlusion_info=occlusion_info
        )
        combined_cost += appearance_weight * appearance_cost
        total_weight += appearance_weight
    
    # IoU component
    if iou_weight > 0:
        track_occlusion = occlusion_info.get('track_occlusion_levels') if occlusion_info else None
        det_occlusion = occlusion_info.get('detection_occlusion_levels') if occlusion_info else None
        
        iou_cost = occlusion_aware_iou_distance(
            tracks, detections, track_occlusion, det_occlusion
        )
        combined_cost += iou_weight * iou_cost
        total_weight += iou_weight
    
    # Motion component
    if motion_weight > 0 and kf is not None:
        # Start with appearance cost as base
        motion_cost = appearance_cost if appearance_weight > 0 else np.ones_like(combined_cost)
        motion_cost = adaptive_fuse_motion(
            kf, motion_cost, tracks, detections, track_qualities=track_qualities
        )
        combined_cost += motion_weight * motion_cost
        total_weight += motion_weight
    
    # Normalize by total weight
    if total_weight > 0:
        combined_cost /= total_weight
    
    return combined_cost


def compute_distance_matrix(tracks, detections, 
                          distance_type='multi_modal',
                          **kwargs):
    """
    Unified interface for computing distance matrices with different methods.
    
    Args:
        tracks: List of track objects
        detections: List of detection objects
        distance_type: Type of distance ('iou', 'embedding', 'multi_modal')
        **kwargs: Additional arguments for specific distance functions
        
    Returns:
        Distance matrix
    """
    if distance_type == 'iou':
        return iou_distance(tracks, detections)
    elif distance_type == 'embedding':
        return enhanced_embedding_distance(tracks, detections, **kwargs)
    elif distance_type == 'multi_modal':
        return multi_modal_distance(tracks, detections, **kwargs)
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


# Backward compatibility aliases
def linear_assignment_with_threshold(cost_matrix, thresh):
    """Backward compatibility alias"""
    return enhanced_linear_assignment(cost_matrix, thresh)
