# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.utils.matching import chi2inv95

INFTY_COST = 1e5


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Enhanced linear assignment solver with improved cost handling.
    
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    
    # Enhanced cost matrix processing
    cost_matrix = _enhance_cost_matrix(cost_matrix, tracks, detections, track_indices, detection_indices)
    
    # Apply distance threshold
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric,
    max_distance,
    cascade_depth,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Enhanced matching cascade with improved stage management.
    
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    
    # Enhanced cascade matching with quality-based prioritization
    # Group tracks by time since update (fresher tracks get priority)
    tracks_by_age = {}
    for track_idx in track_indices:
        age = tracks[track_idx].time_since_update
        if age not in tracks_by_age:
            tracks_by_age[age] = []
        tracks_by_age[age].append(track_idx)
    
    # Process tracks in order of freshness (lower age = higher priority)
    for age in sorted(tracks_by_age.keys()):
        if age > cascade_depth:
            break
        
        track_indices_l = tracks_by_age[age]
        
        # Further prioritize by track quality within same age group
        track_indices_l = _prioritize_tracks_by_quality(tracks, track_indices_l)
        
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            track_indices_l,
            unmatched_detections,
        )
        matches += matches_l
    
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    cost_matrix,
    tracks,
    detections,
    track_indices,
    detection_indices,
    mc_lambda,
    gated_cost=INFTY_COST,
    only_position=False,
):
    """Enhanced cost matrix gating with improved motion consistency integration.
    
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.
    
    Parameters
    ----------
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    mc_lambda : float
        Motion consistency lambda parameter for blending appearance and motion costs.
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """

    gating_threshold = chi2inv95[4]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        
        # Enhanced gating distance computation
        gating_distance = track.kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        
        # Apply gating threshold
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        
        # Enhanced motion consistency integration
        motion_cost = _compute_enhanced_motion_cost(track, detections, detection_indices, gating_distance)
        
        # Adaptive blending based on track quality
        adaptive_lambda = _compute_adaptive_lambda(track, mc_lambda)
        
        # Blend appearance and motion costs
        cost_matrix[row] = (
            adaptive_lambda * cost_matrix[row] + (1 - adaptive_lambda) * motion_cost
        )
        
        # Apply track-specific cost adjustments
        cost_matrix[row] = _apply_track_specific_adjustments(cost_matrix[row], track, detections, detection_indices)
    
    return cost_matrix


def _enhance_cost_matrix(cost_matrix, tracks, detections, track_indices, detection_indices):
    """Apply various enhancements to the cost matrix for better matching"""
    enhanced_matrix = cost_matrix.copy()
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        for col, det_idx in enumerate(detection_indices):
            detection = detections[det_idx]
            
            # Quality-based cost adjustment
            quality_factor = _compute_quality_factor(track, detection)
            enhanced_matrix[row, col] *= quality_factor
            
            # Class consistency bonus
            if hasattr(track, 'cls') and hasattr(detection, 'cls'):
                if track.cls == detection.cls:
                    enhanced_matrix[row, col] *= 0.9  # Small bonus for same class
            
            # Confidence-based adjustment
            conf_factor = _compute_confidence_factor(track, detection)
            enhanced_matrix[row, col] *= conf_factor
    
    return enhanced_matrix


def _prioritize_tracks_by_quality(tracks, track_indices):
    """Prioritize tracks by their quality scores within the same age group"""
    def get_track_priority(track_idx):
        track = tracks[track_idx]
        # Higher quality score = higher priority (lower sort value)
        quality = getattr(track, 'quality_score', 0.5)
        stability = getattr(track, 'stability_score', 0.5)
        return -(quality + stability)  # Negative for descending order
    
    return sorted(track_indices, key=get_track_priority)


def _compute_enhanced_motion_cost(track, detections, detection_indices, gating_distance):
    """Compute enhanced motion cost incorporating multiple motion factors"""
    motion_costs = gating_distance.copy()
    
    # Apply motion consistency from track
    if hasattr(track, 'motion_consistency'):
        motion_factor = 2.0 - track.motion_consistency  # Higher consistency = lower cost
        motion_costs *= motion_factor
    
    # Apply velocity-based prediction cost
    if hasattr(track, 'velocity_history') and len(track.velocity_history) > 0:
        predicted_pos = track.mean[:2] + track.velocity_history[-1]
        for i, det_idx in enumerate(detection_indices):
            detection = detections[det_idx]
            det_pos = detection.to_xyah()[:2]
            velocity_error = np.linalg.norm(det_pos - predicted_pos)
            velocity_factor = 1.0 + min(velocity_error / 50.0, 1.0)  # Normalize to reasonable range
            motion_costs[i] *= velocity_factor
    
    return motion_costs


def _compute_adaptive_lambda(track, base_lambda):
    """Compute adaptive lambda based on track characteristics"""
    adaptive_lambda = base_lambda
    
    # Adjust based on track age (older tracks rely more on motion)
    age_factor = min(track.age / 10.0, 1.0)
    adaptive_lambda = base_lambda + (1 - base_lambda) * age_factor * 0.1
    
    # Adjust based on motion consistency
    if hasattr(track, 'motion_consistency'):
        motion_factor = track.motion_consistency
        adaptive_lambda = adaptive_lambda * (0.8 + 0.4 * motion_factor)
    
    # Adjust based on appearance consistency
    if hasattr(track, 'appearance_consistency'):
        appearance_factor = track.appearance_consistency
        if appearance_factor < 0.5:  # Poor appearance consistency
            adaptive_lambda = min(adaptive_lambda * 1.2, 0.99)  # Rely more on motion
    
    return np.clip(adaptive_lambda, 0.1, 0.99)


def _apply_track_specific_adjustments(costs, track, detections, detection_indices):
    """Apply track-specific cost adjustments"""
    adjusted_costs = costs.copy()
    
    # High-quality track bonus
    if hasattr(track, 'quality_score') and track.quality_score > 0.8:
        adjusted_costs *= 0.95  # Small bonus for high-quality tracks
    
    # Long-term track stability bonus
    if track.hits > 10 and track.time_since_update == 0:
        adjusted_costs *= 0.98  # Bonus for stable long-term tracks
    
    # Recent high-confidence streak bonus
    if hasattr(track, 'high_conf_streak') and track.high_conf_streak > 3:
        adjusted_costs *= 0.97
    
    # Penalize tracks with recent low confidence
    if hasattr(track, 'low_conf_streak') and track.low_conf_streak > 2:
        adjusted_costs *= 1.05
    
    return adjusted_costs


def _compute_quality_factor(track, detection):
    """Compute quality factor for cost adjustment"""
    track_quality = getattr(track, 'quality_score', 0.5)
    detection_quality = getattr(detection, 'quality_score', detection.conf)
    
    # Higher quality leads to lower cost multiplier
    combined_quality = (track_quality + detection_quality) / 2.0
    quality_factor = 1.0 - (combined_quality - 0.5) * 0.2  # Range: 0.9 to 1.1
    
    return np.clip(quality_factor, 0.8, 1.2)


def _compute_confidence_factor(track, detection):
    """Compute confidence-based factor for cost adjustment"""
    # Recent track confidence
    track_conf = getattr(track, 'conf', 0.5)
    detection_conf = detection.conf
    
    # Boost matching for high confidence pairs
    if track_conf > 0.7 and detection_conf > 0.7:
        return 0.9  # Lower cost for high confidence
    elif track_conf < 0.3 or detection_conf < 0.3:
        return 1.1  # Higher cost for low confidence
    else:
        return 1.0  # No adjustment for medium confidence


def _cosine_distance(a, b, data_is_normalized=False):
    """Enhanced cosine distance computation with improved numerical stability.
    
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b = np.asarray(b) / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(a, b.T)
    
    # Ensure numerical stability
    similarity = np.clip(similarity, -1.0, 1.0)
    
    # Convert to distance
    distance = 1.0 - similarity
    
    return distance

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0.0, float(np.inf))
    return r2


def _nn_euclidean_distance(x, y):
    """Enhanced nearest neighbor distance metric (Euclidean) with improved performance.
    
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    if len(x) == 0 or len(y) == 0:
        return np.full(len(y), float('inf'))
    
    distances = _pdist(x, y)
    
    # Use torch for potentially faster computation if available
    if torch.cuda.is_available():
        distances_tensor = torch.from_numpy(distances)
        min_distances = torch.min(distances_tensor, dim=0)[0].numpy()
    else:
        min_distances = np.min(distances, axis=0)
    
    return np.maximum(0.0, min_distances)


def _nn_cosine_distance(x, y):
    """Enhanced nearest neighbor distance metric (cosine) with better handling.
    
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """
    if len(x) == 0 or len(y) == 0:
        return np.full(len(y), float('inf'))
    
    x_ = torch.from_numpy(np.asarray(x))
    y_ = torch.from_numpy(np.asarray(y))
    distances = _cosine_distance(x_, y_)
    
    # Find minimum distance for each query point
    min_distances = distances.min(axis=0)
    
    # Convert to numpy if it's a tensor
    if hasattr(min_distances, 'numpy'):
        min_distances = min_distances.numpy()
    
    return min_distances

class NearestNeighborDistanceMetric(object):
    """
    Enhanced nearest neighbor distance metric with improved memory management
    and adaptive thresholding.
    
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    sample_qualities : Dict[int -> List[float]]
        Quality scores for each sample to enable intelligent sample management.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        self.sample_qualities = {}  # Track quality of samples for better management

    def partial_fit(self, features, targets, active_targets):
        """Enhanced update of the distance metric with quality-based sample management.
        
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            # Initialize if new target
            if target not in self.samples:
                self.samples[target] = []
                self.sample_qualities[target] = []
            
            # Add new sample
            self.samples[target].append(feature)
            
            # Estimate sample quality (could be based on track confidence, consistency, etc.)
            # For now, use a simple quality metric
            quality = np.linalg.norm(feature)  # Feature magnitude as proxy for quality
            self.sample_qualities[target].append(quality)
            
            # Enhanced budget management
            if self.budget is not None and len(self.samples[target]) > self.budget:
                # Remove lowest quality samples instead of just oldest
                samples_with_quality = list(zip(self.samples[target], self.sample_qualities[target]))
                # Sort by quality (descending) and keep the best ones
                samples_with_quality.sort(key=lambda x: x[1], reverse=True)
                samples_with_quality = samples_with_quality[:self.budget]
                
                self.samples[target] = [s for s, q in samples_with_quality]
                self.sample_qualities[target] = [q for s, q in samples_with_quality]
        
        # Keep only active targets but also maintain a small buffer of recent targets
        # for potential ID recovery
        all_targets = set(self.samples.keys())
        inactive_targets = all_targets - set(active_targets)
        
        # Remove old inactive targets but keep some recent ones
        for target in list(inactive_targets):
            # Keep samples for inactive targets for a short period (for ID recovery)
            if len(self.samples[target]) > 0:
                # Reduce budget for inactive targets but don't delete immediately
                max_inactive_samples = min(self.budget // 4, 5) if self.budget else 5
                if len(self.samples[target]) > max_inactive_samples:
                    # Keep only the best samples for inactive targets
                    samples_with_quality = list(zip(self.samples[target], self.sample_qualities[target]))
                    samples_with_quality.sort(key=lambda x: x[1], reverse=True)
                    samples_with_quality = samples_with_quality[:max_inactive_samples]
                    
                    self.samples[target] = [s for s, q in samples_with_quality]
                    self.sample_qualities[target] = [q for s, q in samples_with_quality]

    def distance(self, features, targets):
        """Enhanced distance computation with adaptive thresholding.
        
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if target in self.samples and len(self.samples[target]) > 0:
                cost_matrix[i, :] = self._metric(self.samples[target], features)
            else:
                # No samples available for this target - set high cost
                cost_matrix[i, :] = INFTY_COST
        return cost_matrix

    def get_sample_statistics(self):
        """Get statistics about stored samples for analysis"""
        stats = {}
        for target, samples in self.samples.items():
            stats[target] = {
                'sample_count': len(samples),
                'avg_quality': np.mean(self.sample_qualities[target]) if target in self.sample_qualities else 0,
                'quality_std': np.std(self.sample_qualities[target]) if target in self.sample_qualities and len(self.sample_qualities[target]) > 1 else 0,
            }
        return stats
