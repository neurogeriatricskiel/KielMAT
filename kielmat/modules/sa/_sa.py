import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from kielmat.utils import preprocessing, viz_utils
from typing import Optional

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
    analysis results. This includes plots of vertical acceleration, detected nocturnal rest, posture 
    classifications, and detected turns, offering insights into the dynamics of the analyzed data.

    Methods:
        detect(accel_data, sampling_frequency, tracking_system, tracked_point, plot_results):
            Processes accelerometer data to detect nocturnal rest, classify postures, and identify turning events.
            
        spatio_temporal_parameters():
            Calculates spatial-temporal parameters for detected nocturnal movements.

        metrics():
            Provides overall metrics summarizing the results of the analysis.

    Examples:
        >>> sleep_analyzer = SleepAnalysis()
        >>> sleep_analyzer.detect(
                accel_data=accel_data, 
                sampling_frequency=128,
                tracking_system="imu", 
                tracked_point="LowerBack", 
                plot_results=True
            )
        >>> print(sleep_analyzer.turns_)

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
            min_lying_duration_sec (int): Minimum lying duration in seconds. Default is 300.
        """
        self.lying_threshold = lying_threshold
        self.turn_angle_threshold = turn_angle_threshold
        self.smoothing_window_sec = smoothing_window_sec
        self.sliding_window_sec = sliding_window_sec
        self.overlap_ratio = overlap_ratio
        self.min_lying_duration_sec = min_lying_duration_sec

        # Attributes to store results
        self.nocturnal_rest_ = None
        self.parameters_ = None
        self.metrics_ = None

    def detect(
        self,
        accel_data: np.ndarray,
        sampling_frequency: int,
        tracking_system: Optional[str] = None,
        tracked_point: Optional[str] = None,
        plot_results: bool = False,
    ) -> "SleepAnalysis":
        """
        Detects nocturnal rest and turns based on accelerometer data.

        Args:
            accel_data (np.ndarray): Accelerometer data of shape (N, 3).
            sampling_frequency (int): Sampling frequency in Hz.
            tracking_system (str, optional): Name of the tracking system.
            tracked_point (str, optional): Name of the tracked point.
            plot_results (bool): If True, generates plots of results.

        Returns:
            SleepAnalysis: Instance with detected nocturnal rest stored in `turns_`.

        Example:
            >>> sleep_analyzer = SleepAnalysis()
            >>> sleep_analyzer.detect(accel_data, sampling_frequency=128, plot_results=True)
        """
        if accel_data.shape[1] != 3:
            raise ValueError("Input accelerometer data must have 3 columns (x, y, z).")
        
        # Convert acceleration data from "m/s^2" to "g"
        accel_data /= 9.81
        accel_data = accel_data.to_numpy()

        # Smooth the accelerometer data
        smoothing_window_samples = int(self.smoothing_window_sec * sampling_frequency)
        kernel = np.ones(smoothing_window_samples) / smoothing_window_samples
        smoothed_accel = np.array([
            np.convolve(accel_data[:, i], kernel, mode='same')
            for i in range(accel_data.shape[1])
        ]).T

        # Extract vertical acceleration
        vertical_accel = smoothed_accel[:, 1]

        # Detect nocturnal rest as periods where lying lasts longer than minimum lying duration
        lying_flags = (vertical_accel < self.lying_threshold).astype(int)
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

        # Calculate lying angle (theta) during nocturnal rest for posture analysis
        theta = np.degrees(np.arctan2(smoothed_accel[:, 2], smoothed_accel[:, 0]))

        sliding_window_samples = int(self.sliding_window_sec * sampling_frequency)
        step_size = int(sliding_window_samples * (1 - self.overlap_ratio))
        
        # Determine lying posture based on filtered angle values
        posture = np.zeros(len(accel_data), dtype=int)
        for i in range(0, len(theta) - sliding_window_samples + 1, step_size):
            if nocturnal_rest[i]:
                angle = np.mean(theta[i:i + sliding_window_samples])
                if -45 <= angle <= 45:
                    posture[i:i + sliding_window_samples] = 1  # Back
                elif 45 < angle <= 135:
                    posture[i:i + sliding_window_samples] = 2  # Right side
                elif -135 <= angle < -45:
                    posture[i:i + sliding_window_samples] = 3  # Left side
                else:
                    posture[i:i + sliding_window_samples] = 4  # Belly

        # Detect turning during nocturnal rest
        turn_indices = np.where(np.diff(posture) != 0)[0] # Indices where posture changes
        
        # Extract turn parameters
        turn_start = turn_indices[:-1]
        turn_end = turn_indices[1:]
        turn_duration = (turn_end - turn_start) / sampling_frequency
        turn_angle = theta[turn_end] - theta[turn_start]
        valid_turns = np.abs(turn_angle) >= self.turn_angle_threshold
        turn_duration = turn_duration[valid_turns]
        turn_angle = turn_angle[valid_turns]
        turn_times = turn_end[valid_turns]

        # Store results in DataFrame
        self.turns_ = pd.DataFrame(
            {
                "onset": turn_start[valid_turns] / sampling_frequency,
                "duration": turn_duration,
                "event_type": "turn",
                "tracking_systems": tracking_system,
                "tracked_points": tracked_point,
            }
        )

        # Optional: Generate plots
        if plot_results:
            self._generate_plots(
                vertical_accel, nocturnal_rest, posture,
                theta, turn_times, sampling_frequency
            )
        return self

    def _generate_plots(
        self,
        vertical_accel: np.ndarray,
        nocturnal_rest: np.ndarray,
        posture: np.ndarray,
        theta: np.ndarray,
        turn_times: np.ndarray,
        sampling_frequency: int,
    ):
        """
        Generates plots of accelerometer data, nocturnal rest detection, and postural classification.

        Args:
            vertical_accel (np.ndarray): Vertical acceleration.
            nocturnal_rest (np.ndarray): Binary array indicating rest periods.
            posture (np.ndarray): Postural classification array.
            theta (np.ndarray): Orientation angles.
            turn_times (np.ndarray): Indices of detected turns.
            sampling_frequency (int): Sampling frequency in Hz.
        """
        time = np.arange(len(vertical_accel)) / sampling_frequency

        # Figure
        fig = plt.figure(figsize=(12, 10))

        # Subplot 1: ACCEL data
        ax1 = plt.subplot(211)

        # Plot vertical acceleration (blue) on the left y-axis
        line1, = ax1.plot(time, vertical_accel, label='Vertical Acceleration', color='blue')
        ax1.set_ylabel("Vertical Acceleration (g)", fontsize=14)  # Larger font size for y-axis
        ax1.tick_params(axis='y', labelsize=12)

        # Plot nocturnal rest (detected) with dashed lines, multiplied by 5
        line2, = ax1.plot(time, nocturnal_rest * 5 * np.max(vertical_accel), label='Detected Nocturnal Rest', color='black', linestyle='--')

        # Plot posture classification (red)
        line3, = ax1.plot(time, posture, label='Postural Classification', color='orange', linewidth=2)

        # Adding the Posture Legend/Key in the plot with a similar background box as the plot legend
        ax1.text(0.02, 0.94, "Postures:\n1: Back\n2: Right Side\n3: Left Side\n4: Belly", 
                transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='orange', 
                bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=1'))  
        plt.title("Sleep Analysis", fontsize=16)

        # Create combined legend
        lines = [line1, line2, line3] 
        labels = ['Vertical Acceleration', 'Detected Nocturnal Rest', 'Postural Classification']  
        ax1.legend(lines, labels, loc='upper right', fontsize=12)

        # Subplot 2: Theta (angles) and turns
        ax2 = plt.subplot(212)

        # Plot theta (angles) and detected turns (in the second subplot)
        line_1, = ax2.plot(time, theta, label="Orientation Angle (deg)", color='blue')

        # Plot posture classification
        line_2, = ax2.plot(time, nocturnal_rest * 180 * np.max(vertical_accel), label='Detected Nocturnal Rest', color='black', linestyle='--')
        
        ax2.scatter(time[turn_times], theta[turn_times], 
                    color='red', label='Detected Turns', zorder=5)
        ax2.set_xlabel("Time (s)", fontsize=14) 
        ax2.set_ylabel("Orientation Angle (deg)", fontsize=14)
        lines_ = [line_1, line_2]
        ax2.legend(lines_, labels, loc='upper right', fontsize=12)

        # Plot
        fig.tight_layout()
        plt.show()