# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict

from boxmot.utils.iou import AssociationFunction


def speed_direction_batch(dets, tracks):
    """Compute speed and direction vectors between detections and tracks"""
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def enhanced_linear_assignment(cost_matrix, thresh=None):
    """
    Enhanced linear assignment with better handling of edge cases and numerical stability.
    
    Args:
        cost_matrix (np.ndarray): Cost matrix for assignment
        thresh (float, optional): Cost threshold for valid assignments
        
    Returns:
        tuple: (matches, unmatched_rows, unmatched_cols)
    """
    try:
        import lap
        
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )
        
        # Apply threshold if provided
        if thresh is not None:
            cost_matrix = cost_matrix.copy()
            cost_matrix[cost_matrix > thresh] = thresh + 1e3
        
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, 
                              cost_limit=thresh if thresh is not None else None)
        
        for ix, mx in enumerate(x):
            if mx >= 0:
                if thresh is None or cost_matrix[ix, mx] <= thresh:
                    matches.append([ix, mx])
                else:
                    unmatched_a.append(ix)
        
        # Find unmatched columns
        matched_cols = set([m[1] for m in matches])
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        # Find remaining unmatched rows
        matched_rows = set([m[0] for m in matches])
        unmatched_a.extend([i for i in range(cost_matrix.shape[0]) 
                          if i not in matched_rows and i not in unmatched_a])
        
        matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
        return matches, np.array(unmatched_a), np.array(unmatched_b)
        
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])),
            )
        
        # Handle threshold
        if thresh is not None:
            cost_matrix = cost_matrix.copy()
            cost_matrix[cost_matrix > thresh] = 1e6
        
        x, y = linear_sum_assignment(cost_matrix)
        matches = []
        
        for i, j in zip(x, y):
            if thresh is None or cost_matrix[i, j] <= thresh:
                matches.append([i, j])
        
        # Find unmatched
        matched_rows = set([m[0] for m in matches])
        matched_cols = set([m[1] for m in matches])
        
        unmatched_a = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_b = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
        return matches, np.array(unmatched_a), np.array(unmatched_b)


def linear_assignment(cost_matrix):
    """Legacy function for backward compatibility"""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def compute_occlusion_aware_cost(detections: np.ndarray, 
                               tracks: np.ndarray,
                               det_occlusion_levels: np.ndarray = None,
                               track_occlusion_levels: np.ndarray = None,
                               base_cost_matrix: np.ndarray = None,
                               occlusion_weight: float = 0.1) -> np.ndarray:
    """
    Compute occlusion-aware cost matrix that considers visibility levels.
    
    Args:
        detections: Detection bounding boxes
        tracks: Track bounding boxes
        det_occlusion_levels: Occlusion levels for detections (0=visible, 1=occluded)
        track_occlusion_levels: Occlusion levels for tracks
        base_cost_matrix: Base cost matrix (e.g., from IoU or appearance)
        occlusion_weight: Weight for occlusion penalty
        
    Returns:
        Enhanced cost matrix with occlusion awareness
    """
    if base_cost_matrix is None:
        # Use IoU as base cost if not provided
        iou_matrix = AssociationFunction.iou_batch(detections, tracks)
        base_cost_matrix = 1.0 - iou_matrix
    
    cost_matrix = base_cost_matrix.copy()
    
    # Apply occlusion penalties
    if det_occlusion_levels is not None:
        # Penalize matching with highly occluded detections
        for i, det_occlusion in enumerate(det_occlusion_levels):
            if det_occlusion > 0.5:  # Highly occluded detection
                cost_matrix[i, :] *= (1 + occlusion_weight * det_occlusion)
    
    if track_occlusion_levels is not None:
        # Reduce cost for tracks that should be occluded
        for j, track_occlusion in enumerate(track_occlusion_levels):
            if track_occlusion > 0.3:  # Track is expected to be occluded
                # Reduce cost to maintain track during occlusion
                cost_matrix[:, j] *= (1 - occlusion_weight * track_occlusion * 0.5)
    
    return cost_matrix


def enhanced_associate_detections_to_trackers(detections: np.ndarray,
                                            trackers: np.ndarray,
                                            iou_threshold: float = 0.3,
                                            detection_scores: np.ndarray = None,
                                            track_qualities: np.ndarray = None,
                                            occlusion_info: Dict = None,
                                            use_quality_weighting: bool = True) -> Tuple:
    """
    Enhanced association that considers detection scores, track qualities, and occlusion information.
    
    Args:
        detections: Detection bounding boxes [N, 4]
        trackers: Tracker bounding boxes [M, 4]
        iou_threshold: IoU threshold for association
        detection_scores: Confidence scores for detections [N]
        track_qualities: Quality scores for tracks [M]
        occlusion_info: Dictionary with occlusion information
        use_quality_weighting: Whether to use quality-based weighting
        
    Returns:
        tuple: (matches, unmatched_detections, unmatched_trackers)
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # Compute base IoU matrix
    iou_matrix = AssociationFunction.iou_batch(detections, trackers)
    
    # Create enhanced cost matrix
    cost_matrix = 1.0 - iou_matrix
    
    # Apply quality weighting if enabled
    if use_quality_weighting:
        if detection_scores is not None:
            # Boost high-confidence detections
            det_weight = np.expand_dims(detection_scores, axis=1)
            cost_matrix *= (2.0 - det_weight)  # Lower cost for high confidence
        
        if track_qualities is not None:
            # Boost high-quality tracks
            track_weight = np.expand_dims(track_qualities, axis=0)
            cost_matrix *= (2.0 - track_weight)  # Lower cost for high quality tracks
    
    # Apply occlusion-aware adjustments
    if occlusion_info:
        det_occlusion = occlusion_info.get('detection_occlusion_levels')
        track_occlusion = occlusion_info.get('track_occlusion_levels')
        
        cost_matrix = compute_occlusion_aware_cost(
            detections, trackers, det_occlusion, track_occlusion, 
            cost_matrix, occlusion_weight=0.1
        )
    
    # Perform assignment
    if min(iou_matrix.shape) > 0:
        # Use enhanced assignment with thresholding
        matched_indices, unmatched_det_indices, unmatched_trk_indices = \
            enhanced_linear_assignment(cost_matrix, thresh=1.0 - iou_threshold)
    else:
        matched_indices = np.empty(shape=(0, 2))
        unmatched_det_indices = list(range(len(detections)))
        unmatched_trk_indices = list(range(len(trackers)))

    # Filter matches based on IoU threshold
    if len(matched_indices) > 0:
        valid_matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= iou_threshold:
                valid_matches.append(m)
            else:
                unmatched_det_indices = np.append(unmatched_det_indices, m[0])
                unmatched_trk_indices = np.append(unmatched_trk_indices, m[1])
        
        matched_indices = np.array(valid_matches) if valid_matches else np.empty((0, 2), dtype=int)

    return matched_indices, unmatched_det_indices, unmatched_trk_indices


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Legacy function - Enhanced version for backward compatibility.
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    return enhanced_associate_detections_to_trackers(
        detections, trackers, iou_threshold
    )


def compute_affinity_matrix(detections: np.ndarray,
                          tracks: np.ndarray,
                          appearance_features: np.ndarray = None,
                          track_features: np.ndarray = None,
                          motion_weight: float = 0.7,
                          appearance_weight: float = 0.3,
                          velocities: np.ndarray = None,
                          previous_obs: np.ndarray = None) -> np.ndarray:
    """
    Compute comprehensive affinity matrix combining motion and appearance cues.
    
    Args:
        detections: Detection bounding boxes [N, 4+]
        tracks: Track bounding boxes [M, 4+]
        appearance_features: Detection appearance features [N, D]
        track_features: Track appearance features [M, D]
        motion_weight: Weight for motion component
        appearance_weight: Weight for appearance component
        velocities: Track velocities for motion prediction
        previous_obs: Previous track observations
        
    Returns:
        Affinity matrix [N, M] with combined similarities
    """
    n_dets, n_tracks = len(detections), len(tracks)
    
    if n_dets == 0 or n_tracks == 0:
        return np.zeros((n_dets, n_tracks))
    
    # Motion component (IoU-based)
    iou_matrix = AssociationFunction.iou_batch(detections, tracks)
    motion_affinity = iou_matrix
    
    # Velocity direction consistency (if available)
    if velocities is not None and previous_obs is not None:
        Y, X = speed_direction_batch(detections, previous_obs)
        inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
        diff_angle_cos = inertia_X * X + inertia_Y * Y
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        velocity_consistency = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi
        velocity_consistency = velocity_consistency.T
        
        # Combine with IoU
        motion_affinity = 0.7 * iou_matrix + 0.3 * velocity_consistency
    
    # Appearance component (if available)
    appearance_affinity = np.zeros((n_dets, n_tracks))
    if appearance_features is not None and track_features is not None:
        # Cosine similarity between features
        from scipy.spatial.distance import cdist
        distance_matrix = cdist(appearance_features, track_features, metric='cosine')
        appearance_affinity = 1.0 - distance_matrix
        appearance_affinity = np.clip(appearance_affinity, 0, 1)
    
    # Combine motion and appearance
    if appearance_features is not None and track_features is not None:
        total_affinity = motion_weight * motion_affinity + appearance_weight * appearance_affinity
    else:
        total_affinity = motion_affinity
    
    return total_affinity


def compute_aw_max_metric(emb_cost, w_association_emb, bottom=0.5):
    """
    Enhanced AW-Max metric computation with numerical stability improvements.
    """
    w_emb = np.full_like(emb_cost, w_association_emb)

    # Row-wise normalization
    for idx in range(emb_cost.shape[0]):
        row = emb_cost[idx]
        valid_indices = row > 0  # Only consider positive costs
        if np.sum(valid_indices) < 2:
            continue
            
        valid_costs = row[valid_indices]
        inds = np.argsort(-valid_costs)
        
        if len(inds) < 2:
            continue
            
        max_cost = valid_costs[inds[0]]
        second_max_cost = valid_costs[inds[1]]
        
        if max_cost == 0:
            row_weight = 0
        else:
            ratio = second_max_cost / max_cost
            row_weight = 1 - max((ratio - bottom), 0) / (1 - bottom)
        
        w_emb[idx] *= row_weight

    # Column-wise normalization
    for idj in range(emb_cost.shape[1]):
        col = emb_cost[:, idj]
        valid_indices = col > 0
        if np.sum(valid_indices) < 2:
            continue
            
        valid_costs = col[valid_indices]
        inds = np.argsort(-valid_costs)
        
        if len(inds) < 2:
            continue
            
        max_cost = valid_costs[inds[0]]
        second_max_cost = valid_costs[inds[1]]
        
        if max_cost == 0:
            col_weight = 0
        else:
            ratio = second_max_cost / max_cost
            col_weight = 1 - max((ratio - bottom), 0) / (1 - bottom)
        
        w_emb[:, idj] *= col_weight

    return w_emb * emb_cost


def enhanced_associate(
    detections,
    trackers,
    asso_func,
    iou_threshold,
    velocities=None,
    previous_obs=None,
    vdc_weight=0.1,
    w=1920,
    h=1080,
    emb_cost=None,
    w_assoc_emb=0.5,
    aw_off=False,
    aw_param=0.5,
    detection_scores=None,
    track_qualities=None,
    occlusion_info=None,
    adaptive_threshold=True,
):
    """
    Enhanced association function with occlusion awareness and quality-based matching.
    
    Args:
        detections: Detection bounding boxes
        trackers: Track bounding boxes  
        asso_func: Association function (IoU, GIoU, etc.)
        iou_threshold: IoU threshold for association
        velocities: Track velocities for motion consistency
        previous_obs: Previous track observations
        vdc_weight: Weight for velocity direction consistency
        w, h: Image dimensions
        emb_cost: Embedding cost matrix (appearance)
        w_assoc_emb: Weight for embedding cost
        aw_off: Whether to disable adaptive weighting
        aw_param: Parameter for adaptive weighting
        detection_scores: Detection confidence scores
        track_qualities: Track quality scores
        occlusion_info: Occlusion information dictionary
        adaptive_threshold: Whether to use adaptive thresholding
        
    Returns:
        tuple: (matches, unmatched_detections, unmatched_trackers)
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # Velocity direction consistency
    motion_cost = np.zeros((len(detections), len(trackers)))
    if velocities is not None and previous_obs is not None:
        Y, X = speed_direction_batch(detections, previous_obs)
        inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
        inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
        inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
        diff_angle_cos = inertia_X * X + inertia_Y * Y
        diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
        diff_angle = np.arccos(diff_angle_cos)
        diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

        valid_mask = np.ones(previous_obs.shape[0])
        valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

        motion_cost = (valid_mask * diff_angle) * vdc_weight
        motion_cost = motion_cost.T
        
        # Weight by detection scores if available
        if detection_scores is not None:
            scores = np.repeat(detection_scores[:, np.newaxis], trackers.shape[0], axis=1)
            motion_cost = motion_cost * scores

    # Compute base association cost
    iou_matrix = asso_func(detections, trackers)
    
    # Apply adaptive threshold based on scene complexity
    effective_threshold = iou_threshold
    if adaptive_threshold and occlusion_info:
        crowd_factor = occlusion_info.get('crowd_density', 0.0)
        if crowd_factor > 0.3:  # Crowded scene
            effective_threshold = max(0.1, iou_threshold - 0.1)
    
    # Prepare final cost computation
    total_cost = iou_matrix + motion_cost
    
    # Add appearance cost if available
    if emb_cost is not None:
        emb_cost = emb_cost.copy()
        emb_cost[iou_matrix <= 0] = 0
        
        if not aw_off:
            emb_cost = compute_aw_max_metric(emb_cost, w_assoc_emb, bottom=aw_param)
        else:
            emb_cost *= w_assoc_emb
            
        total_cost += emb_cost
    
    # Apply occlusion-aware adjustments
    if occlusion_info:
        det_occlusion = occlusion_info.get('detection_occlusion_levels')
        track_occlusion = occlusion_info.get('track_occlusion_levels')
        
        if det_occlusion is not None:
            for i, occlusion_level in enumerate(det_occlusion):
                if occlusion_level > 0.5:
                    total_cost[i, :] *= (1 + 0.2 * occlusion_level)
        
        if track_occlusion is not None:
            for j, occlusion_level in enumerate(track_occlusion):
                if occlusion_level > 0.3:
                    total_cost[:, j] *= (1 - 0.1 * occlusion_level)
    
    # Apply quality weighting
    if track_qualities is not None:
        quality_weight = np.expand_dims(track_qualities, axis=0)
        total_cost *= (2.0 - quality_weight)
    
    if detection_scores is not None:
        score_weight = np.expand_dims(detection_scores, axis=1)  
        total_cost *= (2.0 - score_weight)

    # Perform assignment
    if min(iou_matrix.shape):
        a = (iou_matrix > effective_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            final_cost = -total_cost
            matched_indices, unmatched_detections, unmatched_trackers = \
                enhanced_linear_assignment(final_cost, thresh=None)
            
            if matched_indices.size == 0:
                matched_indices = np.empty(shape=(0, 2))
                unmatched_detections = list(range(len(detections)))
                unmatched_trackers = list(range(len(trackers)))
    else:
        matched_indices = np.empty(shape=(0, 2))
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))

    # Validate matches based on IoU threshold
    if len(matched_indices) > 0:
        valid_matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= effective_threshold:
                valid_matches.append(m.reshape(1, 2))
            else:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
        
        if len(valid_matches) == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.concatenate(valid_matches, axis=0)

    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate(
    detections,
    trackers,
    asso_func,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w,
    h,
    emb_cost=None,
    w_assoc_emb=None,
    aw_off=None,
    aw_param=None,
):
    """Enhanced version of the original associate function with backward compatibility"""
    return enhanced_associate(
        detections, trackers, asso_func, iou_threshold,
        velocities, previous_obs, vdc_weight, w, h,
        emb_cost, w_assoc_emb, aw_off, aw_param
    )


def associate_kitti(
    detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight
):
    """
    Enhanced KITTI-style association with category consistency and improved matching.
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    # Velocity direction consistency
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    # IoU computation
    iou_matrix = AssociationFunction.iou_batch(detections, trackers)

    # Category consistency check
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6

    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix

    # Enhanced assignment
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices, unmatched_detections, unmatched_trackers = \
                enhanced_linear_assignment(cost_matrix, thresh=None)
    else:
        matched_indices = np.empty(shape=(0, 2))
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))

    # Filter matches by IoU threshold
    if len(matched_indices) > 0:
        valid_matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= iou_threshold:
                valid_matches.append(m.reshape(1, 2))
            else:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
        
        if len(valid_matches) == 0:
            matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.concatenate(valid_matches, axis=0)

    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)


def multi_stage_association(detections: np.ndarray,
                          tracks: List,
                          high_conf_thresh: float = 0.7,
                          low_conf_thresh: float = 0.3,
                          iou_thresh_high: float = 0.5,
                          iou_thresh_low: float = 0.3,
                          appearance_features: np.ndarray = None,
                          track_features: np.ndarray = None) -> Tuple:
    """
    Multi-stage association for enhanced ID preservation.
    
    Stage 1: High-confidence detections with confirmed tracks
    Stage 2: Medium-confidence detections with remaining tracks  
    Stage 3: IoU-based matching for remaining tracks
    
    Returns:
        tuple: (matches, unmatched_detections, unmatched_tracks)
    """
    if len(tracks) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            []
        )
    
    # Separate detections by confidence
    det_scores = detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
    high_conf_dets = np.where(det_scores >= high_conf_thresh)[0]
    med_conf_dets = np.where((det_scores >= low_conf_thresh) & (det_scores < high_conf_thresh))[0]
    
    # Separate tracks by confirmation status
    confirmed_tracks = [i for i, t in enumerate(tracks) if hasattr(t, 'is_confirmed') and t.is_confirmed()]
    tentative_tracks = [i for i, t in enumerate(tracks) if hasattr(t, 'is_tentative') and t.is_tentative()]
    
    all_matches = []
    remaining_dets = set(range(len(detections)))
    remaining_tracks = set(range(len(tracks)))
    
    # Stage 1: High-confidence detections with confirmed tracks
    if len(high_conf_dets) > 0 and len(confirmed_tracks) > 0:
        stage1_dets = detections[high_conf_dets]
        stage1_tracks = np.array([tracks[i].to_tlwh() if hasattr(tracks[i], 'to_tlwh') 
                                else tracks[i].xyxy for i in confirmed_tracks])
        
        matches_1, unmatched_d1, unmatched_t1 = enhanced_associate_detections_to_trackers(
            stage1_dets, stage1_tracks, iou_thresh_high
        )
        
        # Convert indices back to original
        if len(matches_1) > 0:
            matches_1[:, 0] = high_conf_dets[matches_1[:, 0]]
            matches_1[:, 1] = [confirmed_tracks[i] for i in matches_1[:, 1]]
            all_matches.extend(matches_1.tolist())
            
            remaining_dets -= set(matches_1[:, 0])
            remaining_tracks -= set(matches_1[:, 1])
    
    # Stage 2: Medium-confidence detections with remaining confirmed tracks
    remaining_confirmed = [t for t in confirmed_tracks if t in remaining_tracks]
    if len(med_conf_dets) > 0 and len(remaining_confirmed) > 0:
        stage2_det_indices = [d for d in med_conf_dets if d in remaining_dets]
        if stage2_det_indices:
            stage2_dets = detections[stage2_det_indices]
            stage2_tracks = np.array([tracks[i].to_tlwh() if hasattr(tracks[i], 'to_tlwh') 
                                    else tracks[i].xyxy for i in remaining_confirmed])
            
            matches_2, unmatched_d2, unmatched_t2 = enhanced_associate_detections_to_trackers(
                stage2_dets, stage2_tracks, iou_thresh_low
            )
            
            if len(matches_2) > 0:
                matches_2[:, 0] = [stage2_det_indices[i] for i in matches_2[:, 0]]
                matches_2[:, 1] = [remaining_confirmed[i] for i in matches_2[:, 1]]
                all_matches.extend(matches_2.tolist())
                
                remaining_dets -= set(matches_2[:, 0])
                remaining_tracks -= set(matches_2[:, 1])
    
    # Stage 3: IoU-based matching for all remaining
    if len(remaining_dets) > 0 and len(remaining_tracks) > 0:
        remaining_det_list = list(remaining_dets)
        remaining_track_list = list(remaining_tracks)
        
        stage3_dets = detections[remaining_det_list]
        stage3_tracks = np.array([tracks[i].to_tlwh() if hasattr(tracks[i], 'to_tlwh') 
                                else tracks[i].xyxy for i in remaining_track_list])
        
        matches_3, unmatched_d3, unmatched_t3 = enhanced_associate_detections_to_trackers(
            stage3_dets, stage3_tracks, low_conf_thresh
        )
        
        if len(matches_3) > 0:
            matches_3[:, 0] = [remaining_det_list[i] for i in matches_3[:, 0]]
            matches_3[:, 1] = [remaining_track_list[i] for i in matches_3[:, 1]]
            all_matches.extend(matches_3.tolist())
            
            remaining_dets -= set(matches_3[:, 0])
            remaining_tracks -= set(matches_3[:, 1])
    
    # Format outputs
    final_matches = np.array(all_matches) if all_matches else np.empty((0, 2), dtype=int)
    unmatched_detections = np.array(list(remaining_dets))
    unmatched_tracks = list(remaining_tracks)
    
    return final_matches, unmatched_detections, unmatched_tracks
