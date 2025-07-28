# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os

import numpy as np

from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Enhanced single target track with adaptive state management and confidence-based updates.
    
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    confidence_history : List[float]
        History of detection confidences for adaptive processing.
    quality_score : float
        Overall quality score of the track based on various factors.

    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
    ):
        self.id = id
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha
        self.base_ema_alpha = ema_alpha  # Store original for adaptive adjustment

        # Enhanced state management
        self.state = (
            TrackState.Confirmed
            if (
                os.getenv("GITHUB_ACTIONS") == "true"
                and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark"
            )
            else TrackState.Tentative
        )
        
        # Enhanced feature management
        self.features = []
        self.max_features = 10  # Limit feature history size
        self.confidence_history = [detection.conf]
        self.max_conf_history = 20  # Limit confidence history
        
        if detection.feat is not None:
            detection.feat /= (np.linalg.norm(detection.feat) + 1e-8)
            self.features.append(detection.feat)

        self._n_init = n_init
        self._max_age = max_age
        
        # Track quality and stability metrics
        self.quality_score = getattr(detection, 'quality_score', detection.conf)
        self.stability_score = 0.0
        self.appearance_consistency = 1.0
        self.motion_consistency = 1.0
        
        # Enhanced motion tracking
        self.velocity_history = []
        self.position_history = []
        self.max_motion_history = 10
        
        # Adaptive parameters
        self.missed_detections = 0
        self.confirmed_detections = 1
        self.low_conf_streak = 0
        self.high_conf_streak = 1 if detection.conf > 0.7 else 0

        # Initialize Kalman filter
        self.kf = KalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        
        # Store initial position
        self.position_history.append(self.bbox[:2].copy())

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def camera_update(self, warp_matrix):
        """Enhanced camera motion compensation with stability tracking"""
        [a, b] = warp_matrix
        warp_matrix = np.array([a, b, [0, 0, 1]])
        warp_matrix = warp_matrix.tolist()
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        
        # Store previous position for motion consistency calculation
        prev_pos = self.mean[:2].copy()
        self.mean[:4] = [cx, cy, w / h, h]
        
        # Update motion consistency
        self._update_motion_consistency(prev_pos, np.array([cx, cy]))

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Enhanced Kalman filter prediction with motion analysis"""
        # Store previous state for motion analysis
        prev_mean = self.mean.copy()
        
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
        # Update velocity history
        if len(self.mean) >= 6:
            velocity = self.mean[4:6]
            self.velocity_history.append(velocity.copy())
            if len(self.velocity_history) > self.max_motion_history:
                self.velocity_history.pop(0)
        
        # Update position history
        current_pos = self.mean[:2].copy()
        self.position_history.append(current_pos)
        if len(self.position_history) > self.max_motion_history:
            self.position_history.pop(0)
        
        # Update motion consistency
        if len(self.position_history) >= 2:
            self._update_motion_consistency(self.position_history[-2], current_pos)

    def update(self, detection):
        """Enhanced Kalman filter measurement update with confidence-based processing"""
        # Store previous values for analysis
        prev_conf = self.conf
        prev_features = self.features.copy() if self.features else []
        
        # Update basic attributes
        self.bbox = detection.to_xyah()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        
        # Enhanced Kalman filter update with confidence weighting
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.bbox, self.conf
        )

        # Process feature update with adaptive EMA
        if detection.feat is not None:
            new_feature = detection.feat / (np.linalg.norm(detection.feat) + 1e-8)
            
            if self.features:
                # Calculate appearance consistency
                last_feature = self.features[-1]
                similarity = np.dot(new_feature, last_feature) / (
                    np.linalg.norm(new_feature) * np.linalg.norm(last_feature) + 1e-8
                )
                self.appearance_consistency = 0.9 * self.appearance_consistency + 0.1 * similarity
                
                # Adaptive EMA based on confidence and appearance consistency
                adaptive_alpha = self._calculate_adaptive_ema_alpha(detection.conf, similarity)
                
                # Apply adaptive EMA
                smooth_feat = (
                    adaptive_alpha * last_feature + (1 - adaptive_alpha) * new_feature
                )
                smooth_feat /= (np.linalg.norm(smooth_feat) + 1e-8)
                
                # Store feature with quality-based management
                self.features.append(smooth_feat)
            else:
                # First feature
                self.features.append(new_feature)
            
            # Manage feature history size
            if len(self.features) > self.max_features:
                # Remove oldest features, but keep more recent high-quality ones
                self.features = self.features[-self.max_features:]

        # Update confidence history
        self.confidence_history.append(detection.conf)
        if len(self.confidence_history) > self.max_conf_history:
            self.confidence_history.pop(0)
        
        # Update streak counters
        if detection.conf > 0.7:
            self.high_conf_streak += 1
            self.low_conf_streak = 0
        elif detection.conf < 0.3:
            self.low_conf_streak += 1
            self.high_conf_streak = 0
        else:
            self.low_conf_streak = 0
            self.high_conf_streak = 0

        # Update track metrics
        self.hits += 1
        self.confirmed_detections += 1
        self.time_since_update = 0
        
        # Update quality score
        self._update_quality_score(detection)
        
        # Update stability score
        self._update_stability_score()
        
        # Enhanced state transition logic
        if self.state == TrackState.Tentative:
            # More flexible confirmation based on quality
            if self.hits >= self._n_init or (self.hits >= 1 and self.quality_score > 0.8):
                self.state = TrackState.Confirmed

    def mark_missed(self):
        """Enhanced miss handling with confidence-based deletion logic"""
        self.missed_detections += 1
        
        # More conservative deletion for high-quality tracks
        deletion_threshold = self._max_age
        if self.quality_score > 0.8:
            deletion_threshold = int(self._max_age * 1.5)  # Extend life for high-quality tracks
        elif self.quality_score < 0.3:
            deletion_threshold = int(self._max_age * 0.5)  # Shorter life for low-quality tracks
        
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > deletion_threshold:
            self.state = TrackState.Deleted

    def _calculate_adaptive_ema_alpha(self, detection_conf, appearance_similarity):
        """Calculate adaptive EMA alpha based on detection confidence and appearance consistency"""
        base_alpha = self.base_ema_alpha
        
        # Increase alpha (more smoothing) for low confidence detections
        conf_factor = 1.0 if detection_conf > 0.7 else (0.5 if detection_conf > 0.3 else 0.2)
        
        # Increase alpha for inconsistent appearances
        appearance_factor = 1.0 if appearance_similarity > 0.7 else (0.7 if appearance_similarity > 0.4 else 0.4)
        
        # Combine factors
        adaptive_alpha = base_alpha * conf_factor * appearance_factor
        
        # Ensure reasonable bounds
        adaptive_alpha = np.clip(adaptive_alpha, 0.1, 0.95)
        
        return adaptive_alpha

    def _update_quality_score(self, detection):
        """Update overall track quality score"""
        # Base quality from current confidence
        conf_quality = detection.conf
        
        # Historical confidence quality
        if len(self.confidence_history) > 1:
            avg_conf = np.mean(self.confidence_history)
            conf_stability = 1.0 - np.std(self.confidence_history)
            conf_quality = 0.7 * conf_quality + 0.3 * avg_conf * conf_stability
        
        # Track longevity bonus
        longevity_bonus = min(self.hits / 20.0, 0.2)
        
        # Appearance consistency bonus
        appearance_bonus = max(0, (self.appearance_consistency - 0.5) * 0.2)
        
        # Motion consistency bonus
        motion_bonus = max(0, (self.motion_consistency - 0.5) * 0.1)
        
        # Combine all factors
        self.quality_score = conf_quality + longevity_bonus + appearance_bonus + motion_bonus
        self.quality_score = np.clip(self.quality_score, 0.0, 1.0)

    def _update_stability_score(self):
        """Update track stability score based on various factors"""
        # Confidence stability
        if len(self.confidence_history) > 3:
            conf_stability = 1.0 - min(np.std(self.confidence_history), 1.0)
        else:
            conf_stability = 0.5
        
        # Hit rate
        hit_rate = self.confirmed_detections / max(self.age, 1)
        
        # Consistency scores
        consistency_score = 0.4 * self.appearance_consistency + 0.3 * self.motion_consistency + 0.3 * conf_stability
        
        # Combine factors
        self.stability_score = 0.5 * hit_rate + 0.5 * consistency_score
        self.stability_score = np.clip(self.stability_score, 0.0, 1.0)

    def _update_motion_consistency(self, prev_pos, current_pos):
        """Update motion consistency score"""
        if len(self.velocity_history) >= 2:
            # Calculate predicted position based on previous velocity
            prev_velocity = self.velocity_history[-1] if self.velocity_history else np.zeros(2)
            predicted_pos = prev_pos + prev_velocity
            
            # Calculate actual movement
            actual_movement = current_pos - prev_pos
            predicted_movement = predicted_pos - prev_pos
            
            # Calculate consistency (inverse of prediction error)
            if np.linalg.norm(predicted_movement) > 0:
                error = np.linalg.norm(actual_movement - predicted_movement)
                max_expected_error = max(np.linalg.norm(predicted_movement) * 0.5, 10.0)
                consistency = max(0, 1.0 - (error / max_expected_error))
            else:
                consistency = 1.0 if np.linalg.norm(actual_movement) < 5.0 else 0.5
            
            # Update motion consistency with EMA
            self.motion_consistency = 0.8 * self.motion_consistency + 0.2 * consistency

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def get_track_info(self):
        """Get comprehensive track information for analysis"""
        return {
            'id': self.id,
            'state': self.state,
            'hits': self.hits,
            'age': self.age,
            'time_since_update': self.time_since_update,
            'conf': self.conf,
            'quality_score': self.quality_score,
            'stability_score': self.stability_score,
            'appearance_consistency': self.appearance_consistency,
            'motion_consistency': self.motion_consistency,
            'confirmed_detections': self.confirmed_detections,
            'missed_detections': self.missed_detections,
            'high_conf_streak': self.high_conf_streak,
            'low_conf_streak': self.low_conf_streak,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0,
            'conf_std': np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0,
            'feature_count': len(self.features),
        }

    def can_recover_id(self, similarity_threshold=0.7):
        """Check if this track is suitable for ID recovery"""
        if self.state != TrackState.Deleted:
            return False
        
        # Check if track had good quality
        if self.quality_score < 0.4:
            return False
        
        # Check if track had sufficient history
        if self.hits < 3:
            return False
        
        # Check if features are available
        if not self.features:
            return False
        
        return True

    def compute_appearance_distance(self, features):
        """Compute appearance distance to given features for ID recovery"""
        if not self.features or features is None:
            return float('inf')
        
        # Use the best features (last few with highest quality)
        if len(self.features) > 3:
            # Use last 3 features for more robust matching
            track_features = np.array(self.features[-3:])
            avg_track_feature = np.mean(track_features, axis=0)
        else:
            avg_track_feature = self.features[-1]
        
        # Normalize features
        avg_track_feature = avg_track_feature / (np.linalg.norm(avg_track_feature) + 1e-8)
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Compute cosine distance
        similarity = np.dot(avg_track_feature, features)
        distance = 1.0 - similarity
        
        return distance
