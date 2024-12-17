import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from kielmat.utils import preprocessing, viz_utils

class SleepAnalysis:
    """
    This algorithm implements a sleep analysis system to detect nocturnal rest, classify postures,
    detect turns, and evaluate rest efficiency using accelerometer data collected from an inertial
    measurement unit (IMU) sensor.

    The core of the algorithm lies in the `detect` method, where accelerometer data is processed to 
    identify periods of nocturnal rest, classify body postures, and detect turning events. The process 
    begins with the smoothing of accelerometer signals to reduce noise, followed by the detection of 
    periods of lying based on vertical acceleration thresholds. These periods are further refined by 
    removing short bouts that do not meet a minimum duration criterion.

    Posture classification is performed using the orientation angle (theta), calculated from the 
    accelerometer's mediolateral, anterior-posterior, and vertical components. The algorithm divides 
    postures into four categories: back, right side, left side, and belly. Turns are detected by 
    identifying transitions between these classified postures.

    Detected turns are characterized by their onset times and durations. The algorithm 
    uses a turn angle threshold to filter out minor movements, ensuring that only significant turns are 
    included in the analysis. The results are stored in a pandas DataFrame (`nocturnal_rest_` attribute) 
    in BIDS-like format, containing details such as the onset, duration, and type of each detected event.

    Optionally, if `plot_results` is set to True, the algorithm generates a detailed visualization of the 
    analysis results. This includes plots of vertical acceleration, detected nocturnal rest and posture 
    classifications , offering insights into the dynamics of the analyzed data.

    Methods:
        detect(accel_data, v_accel_col_name, sampling_frequency, tracking_system, tracked_point, plot_results):
            Processes accelerometer data to detect nocturnal rest, classify postures, and identify events.

    Examples:
        >>> sleep_analyzer = SleepAnalysis()
        >>> sleep_analyzer.detect(
                accel_data=accel_data,
                v_accel_col_name="SA_ACCEL_y",
                sampling_frequency=128,
                tracking_system="imu", 
                tracked_point="LowerBack", 
                plot_results=True
            )
        >>> print(sleep_analyzer.posture_)

    References:
        [1] Louter et al. (2015). Accelerometer-based quantitative analysis of axial nocturnal movements ...
        [2] Mirelman et al. (2020). Tossing and Turning in Bed: Nocturnal Movements in Parkinson's Disease ...
    """
    
    def __init__(
        self,
        lying_threshold: float = 0.4,
        turn_angle_threshold: float = 10.0,
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

        # Attributes to store results
        self.nocturnal_rest_ = None

    def detect(
        self,
        accel_data: np.ndarray,
        v_accel_col_name: str,
        sampling_frequency: int,
        tracking_system: Optional[str] = None,
        tracked_point: Optional[str] = None,
        dt_data: Optional[pd.Series] = None,
        plot_results: bool = False,
    ) -> "SleepAnalysis":
        """
        Detects nocturnal rest and turns based on accelerometer data.

        Args:
            accel_data (np.ndarray): Accelerometer data of shape (N, 3).
            v_accel_col_name (str): Column name corresponding to the vertical acceleration.
            sampling_frequency (int): Sampling frequency in Hz.
            tracking_system (str, optional): Name of the tracking system.
            tracked_point (str, optional): Name of the tracked point.
            dt_data (pd.Series, optional): Timestamps corresponding to each sample.
            plot_results (bool): If True, generates plots of results.

        Returns:
            The detected events information is stored in the 'posture_' attribute, which is a pandas DataFrame in BIDS format 

        Example:
            >>> sleep_analyzer = SleepAnalysis()
            >>> sleep_analyzer.detect(accel_data, v_accel_col_name="SA_ACCEL_y", sampling_frequency=100, plot_results=True)
        """
        # Smooth the vertical acceleration signal
        vertical_accel = accel_data[v_accel_col_name].values / 9.81
        smoothing_window_samples = int(self.smoothing_window_sec * sampling_frequency)
        kernel = np.ones(smoothing_window_samples) / smoothing_window_samples
        smoothed_vertical_accel = np.convolve(vertical_accel, kernel, mode="same")

        # Detect lying periods
        lying_flags = (smoothed_vertical_accel < self.lying_threshold).astype(int)
        min_samples = int(self.min_lying_duration_sec * sampling_frequency)
        nocturnal_rest = np.copy(lying_flags)
        start_idx = None
        for i in range(len(nocturnal_rest)):
            if nocturnal_rest[i] == 1 and start_idx is None:
                start_idx = i
            elif nocturnal_rest[i] == 0 and start_idx is not None:
                if (i - start_idx) < min_samples:
                    nocturnal_rest[start_idx:i] = 0
                start_idx = None

        # Identify nocturnal rest start (>min_rest_start_duration_sec lying bout)
        lying_bouts = np.where(np.diff(nocturnal_rest) != 0)[0] + 1
        start_rest = next(
            (idx for idx in lying_bouts
            if np.sum(nocturnal_rest[idx:idx + self.min_rest_start_duration_sec * sampling_frequency])
            >= self.min_rest_start_duration_sec * sampling_frequency),
            None,
        )
        if start_rest is not None:
            nocturnal_rest[:start_rest] = 0  # Ignore anything before the first long lying bout

        # Detect the end of nocturnal rest (first upright >min_rest_interruption_duration_sec)
        last_rest_idx = lying_bouts[-1] if len(lying_bouts) >= 2 else len(nocturnal_rest)
        upright_flags = 1 - nocturnal_rest
        end_rest = next(
            (idx for idx in range(last_rest_idx, len(upright_flags))
            if np.sum(upright_flags[idx:idx + self.min_rest_interruption_duration_sec * sampling_frequency])
            >= self.min_rest_interruption_duration_sec * sampling_frequency),
            None
        )
        if end_rest is not None:
            nocturnal_rest[end_rest:] = 0

        # Compute orientation angle for posture classification
        horizontal_axes = [col for col in accel_data.columns if col != v_accel_col_name]
        acc_h1 = np.convolve(accel_data[horizontal_axes[0]].values / 9.81, kernel, mode="same")
        acc_h2 = np.convolve(accel_data[horizontal_axes[1]].values / 9.81, kernel, mode="same")
        theta = np.degrees(np.arctan2(acc_h2, acc_h1))

        # Posture classification
        sliding_window_samples = int(self.sliding_window_sec * sampling_frequency)
        step_size = int(sliding_window_samples * (1 - self.overlap_ratio))
        posture = np.zeros(len(theta), dtype=int)
        for i in range(0, len(theta) - sliding_window_samples + 1, step_size):
            if nocturnal_rest[i:i + sliding_window_samples].any():
                angle = np.mean(theta[i:i + sliding_window_samples])
                if -45 <= angle <= 45:
                    posture[i:i + sliding_window_samples] = 1  # Back
                elif 45 < angle <= 135:
                    posture[i:i + sliding_window_samples] = 2  # Right
                elif -135 <= angle < -45:
                    posture[i:i + sliding_window_samples] = 3  # Left
                else:
                    posture[i:i + sliding_window_samples] = 4  # Belly

        # Record posture events (BIDS format)
        posture_events = []
        current_posture = posture[0]
        start_idx = 0
        for i in range(1, len(posture)):
            if posture[i] != current_posture or i == len(posture) - 1:
                end_idx = i
                onset_time = dt_data.iloc[start_idx] if dt_data is not None else start_idx / sampling_frequency
                duration_time = (end_idx - start_idx) / sampling_frequency
                posture_events.append({
                    "onset": onset_time,
                    "duration": duration_time,
                    "event_type": {0: "Upright", 1: "Back", 2: "Right", 3: "Left", 4: "Belly"}.get(current_posture, "Unknown"),
                    "tracking_system": tracking_system,
                    "tracked_point": tracked_point,
                })
                current_posture = posture[i]
                start_idx = i

        # Save detected events as a DataFrame
        self.posture_ = pd.DataFrame(posture_events)

        # Optional: Generate plots
        if plot_results:
            viz_utils.plot_sleep_analysis(
                smoothed_vertical_accel,
                nocturnal_rest,
                posture,
                theta,
                sampling_frequency,
                dt_data,
            )
        return self