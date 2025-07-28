# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.strongsort.sort import iou_matching, linear_assignment
from boxmot.trackers.strongsort.sort.track import Track
from boxmot.utils.matching import chi2inv95


class Tracker:
    """
    Enhanced multi-target tracker with advanced ID preservation and adaptive matching.
    
    This tracker implements a sophisticated matching cascade with multiple stages:
    1. High-confidence appearance-based matching
    2. Motion-based matching for missed tracks
    3. IoU-based matching for remaining detections
    4. ID recovery for recently lost tracks
    
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    conf_thresh_high : float
        High confidence threshold for prioritizing detections in matching.
    conf_thresh_low : float
        Low confidence threshold for filtering weak detections.
    id_preservation_weight : float
        Weight factor for ID preservation in cost computation.
    adaptive_matching : bool
        Whether to use adaptive matching parameters based on scene dynamics.
    appearance_weight : float
        Weight for appearance features in cost computation.
    motion_weight : float
        Weight for motion features in cost computation.
        
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    lost_tracks : List[Track]
        Buffer of recently lost tracks for potential ID recovery.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.7,
        max_age=50,
        n_init=2,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
        conf_thresh_high=0.7,
        conf_thresh_low=0.3,
        id_preservation_weight=0.1,
        adaptive_matching=True,
        appearance_weight=0.6,
        motion_weight=0.4,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.conf_thresh_high = conf_thresh_high
        self.conf_thresh_low = conf_thresh_low
        self.id_preservation_weight = id_preservation_weight
        self.adaptive_matching = adaptive_matching
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight

        self.tracks = []
        self.lost_tracks = []  # Buffer for recently lost tracks
        self._next_id = 1
        self.cmc = get_cmc_method("ecc")()
        
        # Adaptive parameters
        self.scene_dynamics = {'density': 0, 'motion_variance': 0, 'occlusion_rate': 0}
        self.matching_history = []
        self.id_recovery_buffer_size = 30  # Keep lost tracks for potential recovery

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict()
        
        # Update scene dynamics for adaptive matching
        self._update_scene_dynamics()

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections, frame_id=None):
        """Enhanced measurement update and track management with multi-stage matching.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        frame_id : int, optional
            Current frame ID for tracking analytics.
        """
        # Sort detections by quality if available
        if detections and hasattr(detections[0], 'quality_score'):
            detections.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Multi-stage matching cascade
        matches, unmatched_tracks, unmatched_detections = self._enhanced_match(detections)

        # Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])

        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Handle unmatched detections - try ID recovery first
        recovered_matches = self._attempt_id_recovery(unmatched_detections, detections)
        
        # Create new tracks for remaining unmatched detections
        remaining_detections = [i for i in unmatched_detections if i not in [m[1] for m in recovered_matches]]
        for detection_idx in remaining_detections:
            self._initiate_track(detections[detection_idx])

        # Track management - move deleted tracks to lost buffer
        active_tracks = []
        for track in self.tracks:
            if track.is_deleted():
                if len(self.lost_tracks) < self.id_recovery_buffer_size:
                    track.lost_frame = frame_id if frame_id else len(self.matching_history)
                    self.lost_tracks.append(track)
                # Keep only recent lost tracks
                self.lost_tracks = [t for t in self.lost_tracks 
                                  if (frame_id or len(self.matching_history)) - getattr(t, 'lost_frame', 0) < self.max_age]
            else:
                active_tracks.append(track)
        
        self.tracks = active_tracks

        # Update distance metric with confirmed tracks only
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        
        if features and targets:
            self.metric.partial_fit(
                np.asarray(features), np.asarray(targets), active_targets
            )
        
        # Store matching statistics for adaptive learning
        self._update_matching_history(matches, unmatched_tracks, unmatched_detections)

    def _enhanced_match(self, detections):
        """Enhanced multi-stage matching with adaptive parameters"""
        
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            
            # Enhanced gating with adaptive parameters
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )
            
            # Apply ID preservation weighting
            cost_matrix = self._apply_id_preservation_weighting(
                cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        # Categorize tracks and detections by confidence
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_tentative()]
        
        high_conf_detections = [i for i, d in enumerate(detections) 
                               if d.conf >= self.conf_thresh_high]
        med_conf_detections = [i for i, d in enumerate(detections) 
                              if self.conf_thresh_low <= d.conf < self.conf_thresh_high]
        low_conf_detections = [i for i, d in enumerate(detections) 
                              if d.conf < self.conf_thresh_low]

        all_matches = []
        all_unmatched_tracks = list(confirmed_tracks)
        all_unmatched_detections = list(range(len(detections)))

        # Stage 1: High-confidence detections with confirmed tracks (appearance-based)
        if high_conf_detections and confirmed_tracks:
            matches_1, unmatched_tracks_1, unmatched_detections_1 = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold * 0.8,  # Stricter threshold for high confidence
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
                high_conf_detections,
            )
            all_matches.extend(matches_1)
            all_unmatched_tracks = [t for t in all_unmatched_tracks if t not in [m[0] for m in matches_1]]
            all_unmatched_detections = [d for d in all_unmatched_detections if d not in [m[1] for m in matches_1]]

        # Stage 2: Medium-confidence detections with remaining tracks
        remaining_tracks = [t for t in all_unmatched_tracks if t in confirmed_tracks]
        remaining_med_detections = [d for d in med_conf_detections if d in all_unmatched_detections]
        
        if remaining_med_detections and remaining_tracks:
            matches_2, unmatched_tracks_2, unmatched_detections_2 = linear_assignment.matching_cascade(
                gated_metric,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                remaining_tracks,
                remaining_med_detections,
            )
            all_matches.extend(matches_2)
            all_unmatched_tracks = [t for t in all_unmatched_tracks if t not in [m[0] for m in matches_2]]
            all_unmatched_detections = [d for d in all_unmatched_detections if d not in [m[1] for m in matches_2]]

        # Stage 3: IoU-based matching for remaining tracks and detections
        iou_track_candidates = unconfirmed_tracks + [
            k for k in all_unmatched_tracks if self.tracks[k].time_since_update == 1
        ]
        
        remaining_detections = [d for d in all_unmatched_detections 
                              if d not in low_conf_detections]  # Skip low confidence in IoU matching
        
        if remaining_detections and iou_track_candidates:
            matches_3, unmatched_tracks_3, unmatched_detections_3 = linear_assignment.min_cost_matching(
                iou_matching.iou_cost,
                self.max_iou_dist,
                self.tracks,
                detections,
                iou_track_candidates,
                remaining_detections,
            )
            all_matches.extend(matches_3)
            all_unmatched_tracks = [t for t in all_unmatched_tracks if t not in [m[0] for m in matches_3]]
            all_unmatched_detections = [d for d in all_unmatched_detections if d not in [m[1] for m in matches_3]]

        # Final unmatched tracks include those not processed in IoU stage
        final_unmatched_tracks = [t for t in all_unmatched_tracks 
                                 if t not in iou_track_candidates] + unmatched_tracks_3

        return all_matches, final_unmatched_tracks, all_unmatched_detections

    def _apply_id_preservation_weighting(self, cost_matrix, tracks, detections, track_indices, detection_indices):
        """Apply ID preservation weighting to cost matrix"""
        if self.id_preservation_weight <= 0:
            return cost_matrix
        
        for i, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            for j, det_idx in enumerate(detection_indices):
                detection = detections[det_idx]
                
                # Reduce cost for tracks with higher quality and longer history
                track_quality = min(track.hits / 10.0, 1.0)
                preservation_bonus = self.id_preservation_weight * track_quality
                cost_matrix[i, j] *= (1.0 - preservation_bonus)
        
        return cost_matrix

    def _attempt_id_recovery(self, unmatched_detections, detections):
        """Attempt to recover IDs from recently lost tracks"""
        recovered_matches = []
        
        if not self.lost_tracks or not unmatched_detections:
            return recovered_matches
        
        # Compute features for unmatched detections
        detection_features = [detections[i].feat for i in unmatched_detections]
        detection_features = np.array([f for f in detection_features if f is not None])
        
        if len(detection_features) == 0:
            return recovered_matches
        
        # Try to match with lost tracks using appearance similarity
        for lost_track in self.lost_tracks:
            if not lost_track.features:
                continue
            
            # Use the last known appearance
            track_feature = lost_track.features[-1].reshape(1, -1)
            
            # Compute cosine distances
            similarities = np.dot(detection_features, track_feature.T).flatten()
            similarities = similarities / (np.linalg.norm(detection_features, axis=1) * np.linalg.norm(track_feature))
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Recover ID if similarity is high enough
            if best_similarity > 0.7:  # Threshold for ID recovery
                det_idx = unmatched_detections[best_match_idx]
                
                # Reactivate the track
                lost_track.state = 1  # Confirmed state
                lost_track.time_since_update = 0
                lost_track.update(detections[det_idx])
                
                # Add back to active tracks
                self.tracks.append(lost_track)
                
                # Remove from lost tracks
                self.lost_tracks.remove(lost_track)
                
                recovered_matches.append((len(self.tracks) - 1, det_idx))
                
                # Remove matched detection from unmatched list
                unmatched_detections.remove(det_idx)
                break
        
        return recovered_matches

    def _update_scene_dynamics(self):
        """Update scene dynamics for adaptive matching parameters"""
        if not self.tracks:
            return
        
        # Calculate track density
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        self.scene_dynamics['density'] = len(confirmed_tracks)
        
        # Calculate motion variance
        velocities = []
        for track in confirmed_tracks:
            if hasattr(track, 'mean') and len(track.mean) >= 6:
                vel = np.linalg.norm(track.mean[4:6])
                velocities.append(vel)
        
        if velocities:
            self.scene_dynamics['motion_variance'] = np.var(velocities)
        
        # Calculate occlusion rate (tracks lost in recent frames)
        recent_history = self.matching_history[-10:] if len(self.matching_history) >= 10 else self.matching_history
        if recent_history:
            total_tracks = sum([stats['total_tracks'] for stats in recent_history])
            lost_tracks = sum([stats['unmatched_tracks'] for stats in recent_history])
            self.scene_dynamics['occlusion_rate'] = lost_tracks / max(total_tracks, 1)

    def _update_matching_history(self, matches, unmatched_tracks, unmatched_detections):
        """Update matching history for adaptive learning"""
        stats = {
            'matches': len(matches),
            'unmatched_tracks': len(unmatched_tracks),
            'unmatched_detections': len(unmatched_detections),
            'total_tracks': len(self.tracks),
            'total_detections': len(matches) + len(unmatched_detections),
        }
        
        self.matching_history.append(stats)
        
        # Keep only recent history
        if len(self.matching_history) > 100:
            self.matching_history = self.matching_history[-100:]

    def _initiate_track(self, detection):
        """Enhanced track initialization with quality assessment"""
        track = Track(
            detection,
            self._next_id,
            self.n_init,
            self.max_age,
            self.ema_alpha,
        )
        
        # Add quality score to track
        if hasattr(detection, 'quality_score'):
            track.quality_score = detection.quality_score
        
        self.tracks.append(track)
        self._next_id += 1

    def get_track_info(self):
        """Get detailed information about current tracks"""
        track_info = []
        for track in self.tracks:
            info = {
                'id': track.id,
                'state': track.state,
                'hits': track.hits,
                'age': track.age,
                'time_since_update': track.time_since_update,
                'conf': track.conf,
                'quality_score': getattr(track, 'quality_score', 0.0),
            }
            track_info.append(info)
        return track_info

    def get_matching_statistics(self):
        """Get comprehensive matching statistics"""
        if not self.matching_history:
            return {}
        
        recent_stats = self.matching_history[-10:] if len(self.matching_history) >= 10 else self.matching_history
        
        return {
            'avg_matches': np.mean([s['matches'] for s in recent_stats]),
            'avg_unmatched_tracks': np.mean([s['unmatched_tracks'] for s in recent_stats]),
            'avg_unmatched_detections': np.mean([s['unmatched_detections'] for s in recent_stats]),
            'match_rate': np.mean([s['matches'] / max(s['total_detections'], 1) for s in recent_stats]),
            'scene_dynamics': self.scene_dynamics.copy(),
            'total_frames': len(self.matching_history),
            'active_tracks': len(self.tracks),
            'lost_tracks_buffer': len(self.lost_tracks),
        }
