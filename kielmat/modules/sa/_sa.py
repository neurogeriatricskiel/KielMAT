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
        """
        self.lying_threshold = lying_threshold
        self.turn_angle_threshold = turn_angle_threshold
        self.smoothing_window_sec = smoothing_window_sec
        self.sliding_window_sec = sliding_window_sec
        self.overlap_ratio = overlap_ratio
        self.min_lying_duration_sec = min_lying_duration_sec

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
            >>> sleep_analyzer.detect(accel_data, v_accel_col_name="vertical_acceleration", sampling_frequency=100, plot_results=True)
        """
        # Extract vertical acceleration and covert it to g
        vertical_accel = accel_data[v_accel_col_name].values / 9.81

        # Smooth the vertical acceleration
        smoothing_window_samples = int(self.smoothing_window_sec * sampling_frequency)
        kernel = np.ones(smoothing_window_samples) / smoothing_window_samples
        smoothed_vertical_accel = np.convolve(vertical_accel, kernel, mode="same")

        # Detect nocturnal rest as periods where lying lasts longer than minimum lying duration
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

        # Dynamically calculate the orientation angle using the other two axes
        horizontal_axes = [col for col in accel_data.columns if col != v_accel_col_name]

        if len(horizontal_axes) != 2:
            raise ValueError("The accelerometer data must have exactly two horizontal axes excluding the vertical axis.")

        # Smooth horizontal acceleration components and convert unit to g
        acc_horizontal_1 = np.convolve(accel_data[horizontal_axes[0]].values / 9.81, kernel, mode="same")
        acc_horizontal_2 = np.convolve(accel_data[horizontal_axes[1]].values / 9.81, kernel, mode="same")

        # Compute orientation angle dynamically
        theta = np.degrees(np.arctan2(acc_horizontal_2, acc_horizontal_1))
        self.theta = theta

        sliding_window_samples = int(self.sliding_window_sec * sampling_frequency)
        step_size = int(sliding_window_samples * (1 - self.overlap_ratio))
        
        # Determine lying posture based on filtered angle values
        posture = np.zeros(len(accel_data), dtype=int)  # Default posture: Upright (0)
        for i in range(0, len(theta) - sliding_window_samples + 1, step_size):
            if np.any(nocturnal_rest[i:i + sliding_window_samples]):
                angle = np.mean(theta[i:i + sliding_window_samples])
                if -45 <= angle <= 45:
                    posture[i:i + sliding_window_samples] = 1  # Back
                elif 45 < angle <= 135:
                    posture[i:i + sliding_window_samples] = 2  # Right side
                elif -135 <= angle < -45:
                    posture[i:i + sliding_window_samples] = 3  # Left side
                elif abs(angle) >= 135:
                    posture[i:i + sliding_window_samples] = 4  # Belly

        # Detect turning during nocturnal rest
        turn_indices = np.where(np.diff(posture) != 0)[0]  # Indices where posture changes

        # Extract turn parameters
        turn_start = turn_indices[:-1]
        turn_end = turn_indices[1:]
        turn_duration = (turn_end - turn_start) / sampling_frequency
        turn_angle = theta[turn_end] - theta[turn_start]
        valid_turns = np.abs(turn_angle) >= self.turn_angle_threshold

        # Map posture indices to descriptive names
        posture_map = {
            0: "Upright",
            1: "Back",
            2: "Right",
            3: "Left",
            4: "Belly",
        }

        # Detect and record posture events
        posture_events = []
        current_posture = posture[0]
        start_idx = 0
        
        for i in range(1, len(posture)):
            if posture[i] != current_posture or i == len(posture) - 1:
                end_idx = i
                if dt_data is not None:
                    onset_time = dt_data.iloc[start_idx]
                    duration_time = (dt_data.iloc[end_idx - 1] - dt_data.iloc[start_idx]).total_seconds()
                else:
                    onset_time = start_idx / sampling_frequency
                    duration_time = (end_idx - start_idx) / sampling_frequency
                    
                posture_events.append({
                    "onset": onset_time,
                    "duration": duration_time,
                    "event_type": posture_map.get(current_posture, "Unknown"),
                    "tracking_system": tracking_system,
                    "tracked_point": tracked_point,
                })
                
                current_posture = posture[i]
                start_idx = i

        # Create a DataFrame for detected posture events
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