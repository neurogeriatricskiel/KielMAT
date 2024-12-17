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
        # Smooth vertical acceleration signal
        vertical_accel = accel_data[v_accel_col_name].values / 9.81
        kernel = np.ones(int(self.smoothing_window_sec * sampling_frequency)) / (
            self.smoothing_window_sec * sampling_frequency
        )
        smoothed_vertical_accel = np.convolve(vertical_accel, kernel, mode="same")

        # Detect Lying and Upright Periods
        window_samples = int(1 * sampling_frequency)
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
            if val == 1 and start is None:
                start = i
            elif val == 0 and start is not None:
                lying_groups.append((start, i - 1))
                start = None
        if start is not None:
            lying_groups.append((start, len(vect_lying) - 1))

        # Identify Nocturnal Rest Period
        min_rest_start_samples = int(self.min_rest_start_duration_sec * sampling_frequency)
        min_interrupt_samples = int(self.min_rest_interruption_duration_sec * sampling_frequency)

        idx_beginning_of_night_rest = next(
            (start for start, end in lying_groups if (end - start) >= min_rest_start_samples), None
        )
        idx_end_of_night_rest = next(
            (end for start, end in reversed(lying_groups) if (end - start) >= min_interrupt_samples), None
        )

        vect_night_rest = np.zeros_like(vect_lying)
        if idx_beginning_of_night_rest is not None and idx_end_of_night_rest is not None:
            vect_night_rest[idx_beginning_of_night_rest:idx_end_of_night_rest] = 1

        # Classify Body Positions During Nocturnal Rest
        horizontal_axes = [col for col in accel_data.columns if col != v_accel_col_name]
        acc_h1 = np.convolve(accel_data[horizontal_axes[0]].values / 9.81, kernel, mode="same")
        acc_h2 = np.convolve(accel_data[horizontal_axes[1]].values / 9.81, kernel, mode="same")
        theta = np.degrees(np.arctan2(acc_h2, acc_h1))

        posture = np.full(len(vect_night_rest), 0)  # Default to Non-Nocturnal (0)

        for start, end in lying_groups:
            if vect_night_rest[start:end].any():  # Regions inside nocturnal rest
                angles = theta[start:end]
                if np.abs(angles).mean() <= 45:
                    posture[start:end] = 1  # Back
                elif np.abs(angles).mean() >= 135:
                    posture[start:end] = 2  # Belly
                elif np.mean(angles) > 45 and np.mean(angles) < 135:
                    posture[start:end] = 3  # Right
                elif np.mean(angles) < -45 and np.mean(angles) > -135:
                    posture[start:end] = 4  # Left
                else:
                    posture[start:end] = 5  # Upright

        # Explicitly mark upright periods during nocturnal rest
        vect_night_upright = vect_night_rest * vect_upright
        posture[vect_night_upright == 1] = 5  # Upright

        # Ensure no remaining unknowns in nocturnal rest
        posture[(vect_night_rest == 1) & (posture == 0)] = 5  # Remaining regions default to Upright

        # Create Posture Event Table
        posture_events = []
        current_posture = posture[0]
        start_idx = 0

        for i in range(1, len(posture)):
            if posture[i] != current_posture or i == len(posture) - 1:
                end_idx = i
                onset_time = dt_data.iloc[start_idx] if dt_data is not None else start_idx / sampling_frequency
                duration = (end_idx - start_idx) / sampling_frequency
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
                    }.get(current_posture, "Non-Nocturnal"),
                    "tracking_system": tracking_system,
                    "tracked_point": tracked_point
                })
                current_posture = posture[i]
                start_idx = i

        # Save results to DataFrame
        self.posture_ = pd.DataFrame(posture_events)

        # Optional Visualization
        if plot_results:
            viz_utils.plot_sleep_analysis(
                smoothed_vertical_accel,
                vect_night_rest,
                self.posture_,
                theta,
                sampling_frequency,
                dt_data,
            )

        return self