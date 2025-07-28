# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

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


class KalmanFilter(object):
    """
    Enhanced Kalman filter for tracking bounding boxes with adaptive parameters
    and confidence-based uncertainty modeling.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model with adaptive noise parameters.
    The bounding box location (x, y, a, h) is taken as direct observation of the 
    state space (linear observation model) with confidence-weighted uncertainty.

    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)

        # Enhanced adaptive uncertainty weights
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model and are now adaptive.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
        # Adaptive parameters
        self._min_std_weight_position = 1.0 / 50  # More confident when needed
        self._max_std_weight_position = 1.0 / 10  # Less confident when needed
        self._min_std_weight_velocity = 1.0 / 320  # More confident when needed
        self._max_std_weight_velocity = 1.0 / 80   # Less confident when needed
        
        # Track motion characteristics for adaptation
        self._velocity_history = []
        self._acceleration_history = []
        self._measurement_history = []
        self._confidence_history = []
        self._max_history_length = 10
        
        # Adaptive noise parameters
        self._adaptive_motion_noise = True
        self._adaptive_observation_noise = True

    def initiate(self, measurement, confidence=1.0):
        """Enhanced track initialization with confidence-based uncertainty.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        confidence : float
            Confidence score of the detection for adaptive initialization.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Enhanced confidence-based initialization uncertainty
        confidence_factor = max(0.1, confidence)  # Avoid extremely low confidence
        position_uncertainty = self._std_weight_position / confidence_factor
        velocity_uncertainty = self._std_weight_velocity / confidence_factor

        std = [
            2 * position_uncertainty * measurement[3],
            2 * position_uncertainty * measurement[3],
            1e-2,
            2 * position_uncertainty * measurement[3],
            10 * velocity_uncertainty * measurement[3],
            10 * velocity_uncertainty * measurement[3],
            1e-5,
            10 * velocity_uncertainty * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        
        # Initialize histories
        self._measurement_history = [measurement.copy()]
        self._confidence_history = [confidence]
        
        return mean, covariance

    def predict(self, mean, covariance, confidence=1.0):
        """Enhanced Kalman filter prediction with adaptive motion noise.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        confidence : float
            Recent confidence level for adaptive noise modeling.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # Update motion characteristics
        self._update_motion_characteristics(mean)
        
        # Adaptive motion noise based on recent motion patterns
        motion_std_pos, motion_std_vel = self._compute_adaptive_motion_noise(mean, confidence)
        
        std_pos = [
            motion_std_pos * mean[3],
            motion_std_pos * mean[3],
            1e-2,
            motion_std_pos * mean[3],
        ]
        std_vel = [
            motion_std_vel * mean[3],
            motion_std_vel * mean[3],
            1e-5,
            motion_std_vel * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        """Enhanced projection with confidence-weighted observation uncertainty.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: float
            Detection confidence for adaptive observation noise.
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # Adaptive observation noise based on confidence and motion consistency
        obs_std_pos = self._compute_adaptive_observation_noise(mean, confidence)
        
        std = [
            obs_std_pos * mean[3],
            obs_std_pos * mean[3],
            1e-1,
            obs_std_pos * mean[3],
        ]

        # Enhanced confidence-based uncertainty
        # Higher confidence = lower uncertainty
        confidence_factor = max(0.1, confidence) if confidence > 0 else 0.5
        std = [(1.5 - confidence_factor) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=0.0):
        """Enhanced Kalman filter correction with confidence weighting.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: float
            Detection confidence for adaptive processing.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # Store measurement and confidence for adaptive learning
        self._measurement_history.append(measurement.copy())
        self._confidence_history.append(confidence)
        
        # Limit history length
        if len(self._measurement_history) > self._max_history_length:
            self._measurement_history.pop(0)
            self._confidence_history.pop(0)
        
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        # Enhanced numerical stability
        try:
            chol_factor, lower = scipy.linalg.cho_factor(
                projected_cov, lower=True, check_finite=False
            )
            kalman_gain = scipy.linalg.cho_solve(
                (chol_factor, lower),
                np.dot(covariance, self._update_mat.T).T,
                check_finite=False,
            ).T
        except scipy.linalg.LinAlgError:
            # Fallback for numerical issues
            kalman_gain = np.dot(
                np.dot(covariance, self._update_mat.T),
                np.linalg.pinv(projected_cov)
            )
        
        innovation = measurement - projected_mean

        # Confidence-weighted innovation
        if confidence > 0:
            innovation_weight = min(confidence * 1.2, 1.0)  # Boost high confidence
            innovation *= innovation_weight

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        
        # Ensure covariance remains positive definite
        new_covariance = self._ensure_positive_definite(new_covariance)
        
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Enhanced gating distance computation with improved numerical stability.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # Enhanced numerical stability for Cholesky decomposition
        try:
            cholesky_factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            # Fallback: add small regularization to diagonal
            regularized_cov = covariance + np.eye(covariance.shape[0]) * 1e-6
            try:
                cholesky_factor = np.linalg.cholesky(regularized_cov)
            except np.linalg.LinAlgError:
                # Ultimate fallback: use pseudo-inverse
                cholesky_factor = np.eye(covariance.shape[0])
        
        d = measurements - mean
        try:
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
        except scipy.linalg.LinAlgError:
            # Fallback computation
            inv_cov = np.linalg.pinv(covariance)
            squared_maha = np.sum(d @ inv_cov * d, axis=1)
        
        return squared_maha

    def _update_motion_characteristics(self, mean):
        """Update motion characteristics for adaptive noise modeling"""
        if len(self._velocity_history) > 0:
            # Calculate acceleration
            current_velocity = mean[4:6]
            prev_velocity = self._velocity_history[-1]
            acceleration = current_velocity - prev_velocity
            
            self._acceleration_history.append(acceleration)
            if len(self._acceleration_history) > self._max_history_length:
                self._acceleration_history.pop(0)
        
        # Store current velocity
        self._velocity_history.append(mean[4:6].copy())
        if len(self._velocity_history) > self._max_history_length:
            self._velocity_history.pop(0)

    def _compute_adaptive_motion_noise(self, mean, confidence):
        """Compute adaptive motion noise based on motion patterns and confidence"""
        base_pos_std = self._std_weight_position
        base_vel_std = self._std_weight_velocity
        
        if not self._adaptive_motion_noise:
            return base_pos_std, base_vel_std
        
        # Adjust based on velocity consistency
        if len(self._velocity_history) >= 3:
            velocity_variance = np.var(self._velocity_history, axis=0)
            velocity_consistency = 1.0 / (1.0 + np.mean(velocity_variance))
            
            # Higher consistency = lower motion noise
            motion_factor = 1.0 + (1.0 - velocity_consistency)
            base_pos_std *= motion_factor
            base_vel_std *= motion_factor
        
        # Adjust based on acceleration patterns
        if len(self._acceleration_history) >= 2:
            acc_variance = np.var(self._acceleration_history, axis=0)
            acc_consistency = 1.0 / (1.0 + np.mean(acc_variance))
            
            # Higher acceleration consistency = lower velocity noise
            acc_factor = 1.0 + (1.0 - acc_consistency) * 0.5
            base_vel_std *= acc_factor
        
        # Adjust based on confidence
        if confidence > 0:
            conf_factor = 1.5 - confidence  # Higher confidence = lower noise
            base_pos_std *= conf_factor
            base_vel_std *= conf_factor
        
        # Apply bounds
        pos_std = np.clip(base_pos_std, self._min_std_weight_position, self._max_std_weight_position)
        vel_std = np.clip(base_vel_std, self._min_std_weight_velocity, self._max_std_weight_velocity)
        
        return pos_std, vel_std

    def _compute_adaptive_observation_noise(self, mean, confidence):
        """Compute adaptive observation noise based on measurement consistency"""
        base_obs_std = self._std_weight_position
        
        if not self._adaptive_observation_noise:
            return base_obs_std
        
        # Adjust based on measurement consistency
        if len(self._measurement_history) >= 3:
            # Compute measurement prediction errors
            errors = []
            for i in range(1, len(self._measurement_history)):
                pred = self._measurement_history[i-1]  # Simple prediction (could be enhanced)
                actual = self._measurement_history[i]
                error = np.linalg.norm(pred[:2] - actual[:2])  # Position error
                errors.append(error)
            
            if errors:
                avg_error = np.mean(errors)
                error_factor = 1.0 + min(avg_error / 50.0, 1.0)  # Normalize error
                base_obs_std *= error_factor
        
        # Adjust based on confidence
        if confidence > 0:
            conf_factor = 1.2 - confidence  # Higher confidence = lower observation noise
            base_obs_std *= conf_factor
        
        # Apply bounds
        obs_std = np.clip(base_obs_std, self._min_std_weight_position, self._max_std_weight_position)
        
        return obs_std

    def _ensure_positive_definite(self, covariance):
        """Ensure covariance matrix remains positive definite"""
        try:
            # Check if matrix is positive definite by attempting Cholesky decomposition
            np.linalg.cholesky(covariance)
            return covariance
        except np.linalg.LinAlgError:
            # Matrix is not positive definite, regularize it
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
            
            # Set minimum eigenvalue to small positive value
            min_eigenval = 1e-6
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            regularized_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return regularized_cov

    def get_motion_statistics(self):
        """Get motion statistics for analysis"""
        stats = {
            'velocity_history_length': len(self._velocity_history),
            'acceleration_history_length': len(self._acceleration_history),
            'measurement_history_length': len(self._measurement_history),
        }
        
        if len(self._velocity_history) > 1:
            velocities = np.array(self._velocity_history)
            stats.update({
                'avg_velocity_magnitude': np.mean(np.linalg.norm(velocities, axis=1)),
                'velocity_variance': np.var(velocities, axis=0).tolist(),
                'velocity_consistency': 1.0 / (1.0 + np.mean(np.var(velocities, axis=0))),
            })
        
        if len(self._acceleration_history) > 0:
            accelerations = np.array(self._acceleration_history)
            stats.update({
                'avg_acceleration_magnitude': np.mean(np.linalg.norm(accelerations, axis=1)),
                'acceleration_variance': np.var(accelerations, axis=0).tolist(),
            })
        
        if len(self._confidence_history) > 0:
            stats.update({
                'avg_confidence': np.mean(self._confidence_history),
                'confidence_std': np.std(self._confidence_history),
                'min_confidence': np.min(self._confidence_history),
                'max_confidence': np.max(self._confidence_history),
            })
        
        return stats

    def reset_adaptive_parameters(self):
        """Reset adaptive parameters to default values"""
        self._velocity_history = []
        self._acceleration_history = []
        self._measurement_history = []
        self._confidence_history = []

    def set_adaptive_mode(self, motion_adaptive=True, observation_adaptive=True):
        """Enable or disable adaptive noise modeling"""
        self._adaptive_motion_noise = motion_adaptive
        self._adaptive_observation_noise = observation_adaptive
