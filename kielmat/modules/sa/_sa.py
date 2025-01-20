import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from kielmat.utils import preprocessing, viz_utils

class SleepAnalysis:
    """
    Detects nocturnal rest, postures, and turning events based on accelerometer data.

    This method processes accelerometer data to detect periods of nocturnal rest,
    classify body postures, and identify turning events. It begins by smoothing the
    vertical acceleration signal to reduce noise, followed by detecting lying and
    upright periods based on a specified threshold. The method then classifies
    postures and identifies turns between different postures.

    Vertical acceleration is critical for this analysis as it represents the component
    of acceleration aligned with the gravitational force. The sensor, typically placed
    on the lower back, measures acceleration along three axes:
        - Vertical (Z-axis): Perpendicular to the ground when standing upright.
        - Mediolateral (X-axis): Side-to-side motion.
        - Anteroposterior (Y-axis): Forward-backward motion.

    By normalizing the vertical acceleration to the gravitational constant (9.81 m/s²),
    the algorithm distinguishes between upright and lying positions. A smoothed
    threshold of 0.4 g (approximately 33° from the horizontal plane) is applied to
    classify lying periods. This threshold was derived from studies on nocturnal
    movement analysis, ensuring accurate posture detection during sleep.

    Optionally, if `plot_results` is set to True, the algorithm generates a detailed 
    visualization of the  analysis results. This includes plots of vertical acceleration, 
    detected nocturnal rest and posture classifications , offering insights into the 
    dynamics of the analyzed data.

    Methods:
        detect(accel_data, v_accel_col_name, sampling_frequency, tracking_system, tracked_point, plot_results):
            Processes accelerometer data to detect nocturnal rest, classify postures, and identify events.

            Returns:
                SleepAnalysis: an instance of the class with the detected events
                stored in the 'posture_' attribute.
    Examples:
        >>> sleep_analyzer = SleepAnalysis()
        >>> sleep_analyzer.detect(
                accel_data=accel_data,
                v_accel_col_name="SA_ACCEL_y",
                sampling_frequency_Hz=128,
                tracking_system="imu", 
                tracked_point="LowerBack", 
                plot_results=True
            )
        >>> print(sleep_analyzer.posture_)

    References:
        [1] Louter et al. (2015). Accelerometer-based quantitative analysis of axial nocturnal movements ... https://doi.org/10.1136/jnnp-2013-306851
        [2] Mirelman et al. (2020). Tossing and Turning in Bed: Nocturnal Movements in Parkinson's Disease ... https://doi.org/10.1002/mds.28006
    """
    
    def __init__(
        self,
        lying_threshold: float = 0.4,
        turn_angle_threshold: float = 15.0,
        smoothing_window_sec: float = 10.0,
        sliding_window_sec: float = 1.0,
        overlap_ratio: float = 0.5,
        min_lying_duration_sec: int = 300, 
        min_rest_start_duration_sec: int = 3600,
        min_rest_interruption_duration_sec: int = 900,
    ):
        """
        Initializes the SleepAnalysis instance with configurable constants.

        Args:
            lying_threshold (float): Threshold for detecting lying position in g. Default is 0.4.
            turn_angle_threshold (float): Minimum angle for turn detection in degrees. Default is 10.0.
            smoothing_window_sec (float): Smoothing window size in seconds. Default is 10.0.
            sliding_window_sec (float): Sliding window size for posture calculation in seconds. Default is 1.0.
            overlap_ratio (float): Overlap ratio for sliding window. Default is 0.5.
            min_lying_duration_sec (int): Minimum lying duration in seconds. Default is 300 (5 minutes).
            min_rest_start_duration_sec (int): Minimum lying duration to mark start of rest in seconds. Default is 3600 (60 min).
            min_rest_interruption_duration_sec (int): Minimum upright duration to mark rest interruption in seconds. Default is 900 (15 min).
        """
        self.lying_threshold = lying_threshold
        self.turn_angle_threshold = turn_angle_threshold
        self.smoothing_window_sec = smoothing_window_sec
        self.sliding_window_sec = sliding_window_sec
        self.overlap_ratio = overlap_ratio
        self.min_lying_duration_sec = min_lying_duration_sec
        self.min_rest_start_duration_sec = min_rest_start_duration_sec
        self.min_rest_interruption_duration_sec = min_rest_interruption_duration_sec

    def detect(
        self,
        accel_data: pd.DataFrame,
        v_accel_col_name: str,
        sampling_frequency_Hz: float,
        tracking_system: Optional[str] = None,
        tracked_point: Optional[str] = None,
        dt_data: Optional[pd.Series] = None,
        plot_results: bool = False,
    ) -> pd.DataFrame:
        """
        Detects nocturnal rest and turns based on accelerometer data.

        Args:
            accel_data (np.ndarray): Accelerometer data of shape (N, 3).
            v_accel_col_name (str): Column name corresponding to the vertical acceleration.
            sampling_frequency_Hz (int): Sampling frequency in Hz.
            tracking_system (str, optional): Name of the tracking system.
            tracked_point (str, optional): Name of the tracked point.
            dt_data (pd.Series, optional): Timestamps corresponding to each sample.
            plot_results (bool): If True, generates plots of results.

        Returns:
            The posture information is stored in the 'posture_' attribute, which is a pandas 
            DataFrame in BIDS format with the following information:

                - onset: Start time of the posture.
                - duration: Duration of the posture.
                - event_type: Type of the event (posture type).
                - tracking_systems: Name of the tracking systems.
                - tracked_points: Name of the tracked points on the body.       
        """
        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None:
            # Ensure dt_data is a pandas Series
            if not isinstance(dt_data, pd.Series):
                raise ValueError("dt_data must be a pandas Series with datetime values")

            # Ensure dt_data has datetime values
            if not pd.api.types.is_datetime64_any_dtype(dt_data):
                raise ValueError("dt_data must be a pandas Series with datetime values")

            # Ensure dt_data has the same length as input data
            if len(dt_data) != len(accel_data):
                raise ValueError(
                    "dt_data must be a series with the same length as data"
                )
        
        # Check if data is a DataFrame
        if not isinstance(accel_data, pd.DataFrame):
            raise ValueError("Acceleration data must be a pandas DataFrame")

        # Check if plot_results is a boolean
        if not isinstance(plot_results, bool):
            raise ValueError("plot_results must be a boolean value")

        # Convert acceleration data from m/s² to g (divide by 9.81)
        accel_data = accel_data / 9.81
        
        # Smooth vertical acceleration signal
        vertical_accel = accel_data[v_accel_col_name].values
        kernel = np.ones(int(self.smoothing_window_sec * sampling_frequency_Hz)) / (
            self.smoothing_window_sec * sampling_frequency_Hz
        )
        smoothed_vertical_accel = np.convolve(vertical_accel, kernel, mode="same")
        
        # Determine upright and lying periods based on the threshold
        vect_upright = smoothed_vertical_accel >= self.lying_threshold # Mark upright periods
        vect_lying = ~vect_upright

        # Detect Lying and Upright Periods
        window_samples = int(1 * sampling_frequency_Hz)
        stride = int(window_samples * (1 - self.overlap_ratio))
        vect_upright = np.zeros_like(smoothed_vertical_accel)

        for i in range(0, len(smoothed_vertical_accel) - window_samples, stride):
            mean_acc = np.mean(np.abs(smoothed_vertical_accel[i:i + window_samples]))
            if mean_acc >= self.lying_threshold:
                vect_upright[i:i + window_samples] = 1

        vect_lying = 1 - vect_upright

        # Group Lying Bout
        lying_groups = []
        start = None
        for i, val in enumerate(vect_lying):
            if val and start is None:
                start = i
            elif not val and start is not None:
                if i - start >= self.min_lying_duration_sec * sampling_frequency_Hz:
                    lying_groups.append((start, i - 1))
                start = None
        if start is not None and len(vect_lying) - start >= self.min_lying_duration_sec * sampling_frequency_Hz:
            lying_groups.append((start, len(vect_lying) - 1))

        # Identify Nocturnal Rest Period
        min_rest_start_samples = int(self.min_rest_start_duration_sec * sampling_frequency_Hz)
        min_interrupt_samples = int(self.min_rest_interruption_duration_sec * sampling_frequency_Hz)

        idx_start = next((start for start, end in lying_groups if (end - start) >= min_rest_start_samples), None)
        idx_end = next((end for start, end in reversed(lying_groups) if (end - start) >= min_interrupt_samples), None)
        
        # Identify Nocturnal Rest Period
        vect_night_rest = np.zeros(len(vect_lying), dtype=bool)
        if idx_start is not None and idx_end is not None:
            vect_night_rest[idx_start:idx_end + 1] = True

        # Filter for nocturnal rest periods
        rest_indices = np.where(vect_night_rest)[0]
        if len(rest_indices) == 0:
            raise ValueError("No nocturnal rest periods detected.")

        # Classify Body Positions During Nocturnal Rest
        horizontal_axes = [col for col in accel_data.columns if col != v_accel_col_name]
        self.acc_h1 = np.convolve(accel_data[horizontal_axes[0]].values, kernel, mode="same")
        self.acc_h2 = np.convolve(accel_data[horizontal_axes[1]].values, kernel, mode="same")
        theta = np.degrees(np.arctan2(self.acc_h2, self.acc_h1))

        # Classify Body Positions During Nocturnal Rest
        posture = np.zeros(len(theta), dtype=int)
        previous_angle = theta[rest_indices[0]]
        for i in rest_indices:
            angle = theta[i]
            # Check if vertical acceleration exceeds lying threshold
            if smoothed_vertical_accel[i] >= self.lying_threshold:
                posture[i] = 5  # Upright
            else:
                # Determine lying posture based on angle thresholds
                if abs(angle - previous_angle) >= self.turn_angle_threshold or posture[i - 1] == 5:
                    if -45 <= angle <= 45:
                        posture[i] = 1  # Back
                    elif 135 <= angle or angle <= -135:
                        posture[i] = 2  # Belly
                    elif 45 < angle < 135:
                        posture[i] = 3  # Right
                    elif -135 < angle < -45:
                        posture[i] = 4  # Left
                    previous_angle = angle
                else:
                    posture[i] = posture[i - 1]  # Maintain previous lying posture

        # Create posture events only within nocturnal rest
        posture_events = []
        current_posture = posture[rest_indices[0]]
        start_idx = rest_indices[0]

        for idx in range(1, len(rest_indices)):
            i = rest_indices[idx]
            if posture[i] != current_posture or idx == len(rest_indices) - 1:
                end_idx = i
                onset_time = dt_data.iloc[start_idx] if dt_data is not None else start_idx / sampling_frequency_Hz
                duration = (end_idx - start_idx) / sampling_frequency_Hz
                posture_events.append({
                    "onset": onset_time,
                    "duration": duration,
                    "event_type": {
                        0: "Non-Nocturnal",
                        1: "Back",
                        2: "Belly",
                        3: "Right",
                        4: "Left",
                        5: "Upright"
                    }.get(current_posture, "Unknown"),
                    "tracking_system": tracking_system,
                    "tracked_point": tracked_point,
                })
                current_posture = posture[i]
                start_idx = i

        self.posture_ = pd.DataFrame(posture_events)

        # Optional Visualization
        if plot_results:
            viz_utils.plot_sleep_analysis(
                smoothed_vertical_accel,
                vect_night_rest,
                self.posture_,
                theta,
                sampling_frequency_Hz,
                dt_data,
            )

        return self