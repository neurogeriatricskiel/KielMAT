# Import libraries
import numpy as np
import pandas as pd
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.utils import viz_utils
from typing import Optional, TypeVar

Self = TypeVar("Self", bound="PhamPosturalTransitionDetection")

class PhamPosturalTransitionDetection:
    """
    This algorithm aims to detect postural transitions (e.g., sit to stand or stand to sit movements)
    using accelerometer and gyroscope data collected from a lower back inertial measurement unit (IMU)
    sensor.

    The algorithm is designed to be robust in detecting postural transitions using inertial sensor data 
    and provides detailed information about these transitions. It starts by loading the accelerometer and 
    gyro data, which includes three columns corresponding to the acceleration and gyro signals across 
    the x, y, and z axes, along with the sampling frequency of the data. It first checks the validity of 
    the input data. Then, it calculates the sampling period, selects accelerometer and gyro data. Tilt angle 
    estimation is performed using gyro data in lateral or anteroposterior direction which represent movements 
    or rotations in the mediolateral direction. The tilt angle is decomposed using wavelet transformation to 
    identify stationary periods. Stationary periods are detected using accelerometer variance and gyro variance. 
    Then, peaks in the wavelet-transformed tilt signal are detected as potential postural transition events.

    If there's enough stationary data, further processing is done to estimate the orientation using
    quaternions and to identify the beginning and end of postural transitions using gyro data. Otherwise,
    if there's insufficient stationary data, direction changes in gyro data are used to infer postural
    transitions. Finally, the detected postural transitions along with their characteristics (onset, duration, etc.) 
    are stored in a pandas DataFrame (postural_transitions_ attribute).
    
    In addition, spatial-temporal parameters are calculated using detected postural transitions and their 
    characteristics by the spatio_temporal_parameters method. As a return, the postural transition id along 
    with its spatial-temporal parameters including type of postural transition (sit to stand or stand to sit), 
    angle of postural transition, maximum flexion velocity, and maximum extension velocity are stored in a pandas 
    DataFrame (parameters_ attribute).
    
    If requested (plot_results set to True), it generates plots of the accelerometer and gyroscope data
    along with the detected postural transitions.

    Methods:
        detect(data, gyro_vertical, accel_unit, gyro_unit, sampling_freq_Hz, dt_data, tracking_system, tracked_point, plot_results):
            Detects  sit to stand and stand to sit using accelerometer and gyro signals.

            Args:
                data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
                gyro_mediolateral (str): It corresponds to the gyroscope component representing movements or rotations in the mediolateral direction. This could correspond to rotation around the y-axis in your coordinate system.
                accel_unit (str): Unit of acceleration data.
                gyro_unit (str): Unit of gyro data.
                sampling_freq_Hz (float, int): Sampling frequency of the signals.
                dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
                tracking_system (str, optional): Tracking systems.
                tracked_point (str, optional): Tracked points on the body.
                plot_results (bool, optional): If True, generates a plot. Default is False.

            Returns:
                PhamPosturalTransitionDetection: an instance of the class with the detected postural transitions
                stored in the 'postural_transitions_' attribute.

        spatio_temporal_parameters():
            Extracts spatio-temporal parameters of the detected turns.

            Returns:
                pd.DataFrame: The spatio-temporal parameter information is stored in the 'spatio_temporal_parameters'
                attribute with the following columns:
                    - type_of_postural_transition: Type of postural transition which is either "sit to stand" or "stand to sit".
                    - angel_of_postural_transition: Angle of the postural transition in degrees.
                    - maximum_flexion_velocity: Maximum flexion velocity in deg/s.
                    - maximum_extension_velocity: Maximum extension velocity deg/s.

    Examples:
        >>> pham = PhamPosturalTransitionDetection()
        >>> pham.detect(
                data=input_data,
                gyro_mediolateral="pelvis_ANGVEL_y",
                accel_unit="g",
                gyro_unit="deg/s",
                sampling_freq_Hz=200.0,
                tracking_system="imu",
                tracked_point="LowerBack",
                plot_results=False
                )
        >>> print(pham.postural_transitions_)
                onset      duration     event_type              tracking_systems    tracked_points
            0   17.895     1.8          postural transition     imu                 LowerBack
            1   54.655     1.9          postural transition     imu                 LowerBack

        >>> pham.spatio_temporal_parameters()
        >>> print(pham.parameters_)
                type of postural transition    angle of postural transition     maximum flexion velocity    maximum extension velocity
            0   sit to stand                   53.26                            79                          8
            1   stand to sit                   47.12                            91                      120

    References:
        [1] Pham et al. (2018). Validation of a Lower Back "Wearable"-Based Sit-to-Stand and Stand-to-Sit Algorithm...
    """

    def __init__(
        self,
        cutoff_freq_hz: float = 5.0,
        thr_accel_var: float = 0.05,
        thr_gyro_var: float = 10e-2,
        min_postural_transition_angle_deg: float = 15.0,
    ):
        """
        Initializes the PhamPosturalTransitionDetection instance.

        Args:
            cutoff_freq_hz (float, optional): Cutoff frequency for low-pass Butterworth filer. Default is 5.0.
            thr_accel_var (float): Threshold value for identifying periods where the acceleartion variance is low. Default is 0.5.
            thr_gyro_var (float): Threshold value for identifying periods where the gyro variance is low. Default is 2e-4.
            min_turn_angle_deg (float): Minimum angle which is considered as postural transition in degrees. Default is 15.0.
        """
        self.cutoff_freq_hz=cutoff_freq_hz
        self.thr_accel_var=thr_accel_var
        self.thr_gyro_var=thr_gyro_var
        self.min_postural_transition_angle_deg=min_postural_transition_angle_deg

    def detect(
        self, 
        data: pd.DataFrame,
        gyro_mediolateral: str,
        accel_unit: str, 
        gyro_unit: str, 
        sampling_freq_Hz: float,
        dt_data: Optional[pd.Series] = None,
        tracking_system: Optional[str] = None,
        tracked_point: Optional[str] = None,
        plot_results: bool = False
    ) -> Self:
        """
        Detects postural transitions based on the input accelerometer and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            gyro_mediolateral (str): It corresponds to the gyroscope component representing movements or rotations in the mediolateral direction. This could correspond to rotation around the y-axis in your coordinate system.
            accel_unit (str): Unit of acceleration data.
            gyro_unit (str): Unit of gyro data.
            sampling_freq_Hz (float): Sampling frequency of the input data.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
            tracking_system (str, optional): Tracking systems.
            tracked_point (str, optional): Tracked points on the body.
            plot_results (bool, optional): If True, generates a plot. Default is False.

        Returns:
            The postural transition information is stored in the 'postural_transitions_' attribute,
            which is a pandas DataFrame in BIDS format with the following columns:
                - onset: Start time of the postural transition in second.
                - duration: Duration of the postural transition in second.
                - event_type: Type of the event which is postural transition.
                - tracking_systems: Name of the tracking systems.
                - tracked_points: Name of the tracked points on the body.
        """
        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (
            not isinstance(dt_data, pd.Series)
            or not pd.api.types.is_datetime64_any_dtype(dt_data)
        ):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        # Error handling for invalid input data
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 6:
            raise ValueError(
                "Input accelerometer and gyro data must be a DataFrame with 6 columns for x, y, and z axes."
            )

        # Check if sampling frequency is positive
        if sampling_freq_Hz <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Check if plot_results is a boolean
        if not isinstance(plot_results, bool):
            raise ValueError("plot_results must be a boolean value")

        # Calculate sampling period
        sampling_period = 1 / sampling_freq_Hz

        # Select acceleration data and convert it to numpy array format
        accel = data.iloc[:, 0:3].copy()
        accel = accel.to_numpy()

        # Select gyro data and convert it to numpy array format
        gyro = data.iloc[:, 3:6].copy()
        self.gyro = gyro.to_numpy()

        # Extract vertical accelerometer data using the specified index
        self.gyro_mediolateral = data[gyro_mediolateral].to_numpy()

        # Check unit of acceleration data if it is in g or m/s^2 (including variations)
        if accel_unit in ["m/s^2", "meters/s^2", "meter/s^2"]:
            # Convert acceleration data from m/s^2 to g (if not already is in g)
            accel /= 9.81
        elif accel_unit in ["g", "G"]:
            pass  # No conversion needed
        else:
            raise ValueError("Unsupported unit for acceleration data")

        # Check unit of gyro data if it is in deg/s or rad/s
        if gyro_unit in ["rad/s", "radians per second"]:
            # Convert gyro data from rad/s to deg/s (if not already is in deg/s)
            gyro = np.rad2deg(gyro)
        elif gyro_unit in ["deg/s", "degrees per second"]:
            pass  # Gyro data is already in rad/s
        else:
            raise ValueError("Invalid unit for gyro data. Must be 'deg/s' or 'rad/s'")

        # Calculate timestamps to use in next calculation
        time = np.arange(1, len(accel[:, 0]) + 1) * sampling_period

        # Estimate tilt angle in deg
        tilt_angle_deg = preprocessing.tilt_angle_estimation(
            data=self.gyro_mediolateral, sampling_frequency_hz=sampling_freq_Hz
        )

        # Convert tilt angle to rad
        tilt_angle_rad = np.deg2rad(tilt_angle_deg)

        # Calculate sine of the tilt angle in radians
        tilt_sin = np.sin(tilt_angle_rad)

        # Apply wavelet decomposition with level of 3
        tilt_dwt_3 = preprocessing.wavelet_decomposition(
            data=tilt_sin, level=3, wavetype="coif5"
        )

        # Apply wavelet decomposition with level of 10
        tilt_dwt_10 = preprocessing.wavelet_decomposition(
            data=tilt_sin, level=10, wavetype="coif5"
        )

        # Calculate difference
        tilt_dwt = tilt_dwt_3 - tilt_dwt_10

        # Find peaks in denoised tilt signal 
        # Peaks of the tilt_dwt signal with magnitude and prominence >0.2 were defined as postural transition events
        # Separate positive and negative peak detection
        positive_peaks, _ = scipy.signal.find_peaks(tilt_dwt, height=-0.2, prominence=0.2)
        negative_peaks, _ = scipy.signal.find_peaks(tilt_dwt, height=0.2, prominence=0.2)

        # Merge, sort, and eliminate redundant peaks
        self.local_peaks = np.unique(np.concatenate((positive_peaks, negative_peaks)))

        # Calculate the norm of acceleration
        accel_norm = np.sqrt(
            accel[:, 0] ** 2 + accel[:, 1] ** 2 + accel[:, 2] ** 2
        )

        # Calculate absolute value of the acceleration signal
        accel_norm = np.abs(accel_norm)

        # Detect stationary parts of the signal based on the deifned threshold
        stationary_1 = accel_norm < self.thr_accel_var
        stationary_1 = (stationary_1).astype(int)

        # Compute the variance of the moving window acceleration
        accel_var = preprocessing.moving_var(
            data=accel_norm, window=sampling_freq_Hz
        )

        # Calculate stationary_2 from acceleration variance
        stationary_2 = (accel_var <= self.thr_gyro_var).astype(int)

        # Convert unit of the gyro data from deg/s to rad/s
        gyro = np.deg2rad(self.gyro)

        # Calculate stationary of gyro variance
        gyro_norm = np.sqrt(self.gyro[:, 0] ** 2 + self.gyro[:, 1] ** 2 + self.gyro[:, 2] ** 2)

        # Compute the variance of the moving window of gyro
        gyro_var = preprocessing.moving_var(data=gyro_norm, window=sampling_freq_Hz)

        # Calculate stationary of gyro variance
        stationary_3 = (gyro_var <= self.thr_gyro_var).astype(int)

        # Perform stationarity checks
        stationary = stationary_1 & stationary_2 & stationary_3

        # Remove consecutive True values in the stationary array
        for i in range(len(stationary) - 1):
            if stationary[i] == 1:
                if stationary[i + 1] == 0:
                    stationary[i] = 0

        # Set initial period and check if enough stationary data is available
        # Stationary periods are defined as the episodes when the lower back of the participant was almost not moving and not rotating.
        # The thresholds are determined based on the training data set.
        init_period = 0.1

        if (
            np.sum(stationary[: int(sampling_freq_Hz * init_period)]) >= 200
        ):  # If the process is stationary in the first 2s
            # If there is enough stationary data, perform sensor fusion using accelerometer and gyro data
            postural_transition_onset, postural_transition_type, self.postural_transition_angle, duration, self.flexion_max_vel, self.extension_max_vel = (
                preprocessing.process_postural_transitions_stationary_periods(
                    time,
                    accel,
                    self.gyro,
                    stationary,
                    tilt_angle_deg,
                    sampling_period,
                    sampling_freq_Hz,
                    init_period,
                    self.local_peaks,
                )
            )

            # Create a DataFrame with postural transition information
            postural_transitions_ = pd.DataFrame(
                {
                    "onset": postural_transition_onset,
                    "duration": duration,
                    "event_type": postural_transition_type,
                    "tracking_systems": tracking_system,
                    "tracked_points": tracked_point,
                }
            )

        else:
            # Handle cases where there is not enough stationary data
            # Find indices where the product of consecutive changes sign, indicating a change in direction
            iZeroCr = np.where((self.gyro_mediolateral[:-1] * self.gyro_mediolateral[1:]) < 0)[0]

            # Calculate the difference between consecutive values
            gyro_y_diff = np.diff(self.gyro_mediolateral, axis=0)

            # Initialize arrays to store left and right indices for each local peak
            # Initialize left side indices with ones
            self.left_side = np.ones_like(self.local_peaks)

            # Initialize right side indices with length of gyro data
            self.lef_side = len(self.gyro_mediolateral) * np.ones_like(self.local_peaks)

            # Loop through each local peak
            for i in range(len(self.local_peaks)):
                # Get the index of the current local peak
                postural_transitions = self.local_peaks[i]

                # Calculate distances to all zero-crossing points relative to the peak
                dist2peak = iZeroCr - postural_transitions

                # Extract distances to zero-crossing points on the left side of the peak
                dist2peak_left_side = dist2peak[dist2peak < 0]

                # Extract distances to zero-crossing points on the right side of the peak
                dist2peak_right_side = dist2peak[dist2peak > 0]

                # Iterate over distances to zero-crossing points on the left side of the peak (in reverse order)
                for j in range(len(dist2peak_left_side) - 1, -1, -1):
                    # Check if slope is down and the left side not too close to the peak (more than 200ms)
                    if gyro_y_diff[postural_transitions + dist2peak_left_side[j]] < 0 and -dist2peak_left_side[j] > (0.2*sampling_freq_Hz):
                        if j > 0:
                            # If the left side peak is far enough or small enough
                            if (dist2peak_left_side[j] - dist2peak_left_side[j - 1]) >= (0.2*sampling_freq_Hz) or (
                                tilt_angle_deg[postural_transitions + dist2peak_left_side[j - 1]]
                                - tilt_angle_deg[postural_transitions + dist2peak_left_side[j]]
                            ) > 1:
                                # Store the index of the left side
                                self.left_side[i] = postural_transitions + dist2peak_left_side[j]
                                break
                            else:
                                self.left_side[i] = postural_transitions + dist2peak_left_side[j]
                                break
                for j in range(len(dist2peak_right_side)):
                    if gyro_y_diff[postural_transitions + dist2peak_right_side[j]] < 0 and dist2peak_right_side[j] > (0.2*sampling_freq_Hz):
                        self.lef_side[i] = postural_transitions + dist2peak_right_side[j]
                        break

            # Calculate postural transition angle
            self.postural_transition_angle = np.abs(tilt_angle_deg[self.local_peaks] - tilt_angle_deg[self.left_side])
            if self.left_side[0] == 1:
                self.postural_transition_angle[0] = abs(
                    tilt_angle_deg[self.local_peaks[0]] - tilt_angle_deg[self.lef_side[0]]
                )

            # Calculate duration of each postural transition
            duration = (self.lef_side - self.left_side) / sampling_freq_Hz

            # Convert peak times to integers
            postural_transition_onset = time[self.local_peaks]

        # Remove too small postural transitions
        i = self.postural_transition_angle >= self.min_postural_transition_angle_deg
        postural_transition_onset = self.left_side[i]/ sampling_freq_Hz

        # Create a DataFrame with postural transition information
        postural_transitions_ = pd.DataFrame(
            {
                "onset": postural_transition_onset,
                "duration": duration,
                "event_type": "postural transition",
                "tracking_systems": tracking_system,
                "tracked_points": tracked_point,
            }
        )

        # Assign the DataFrame to the 'postural_transitions_' attribute
        self.postural_transitions_ = postural_transitions_

        # If Plot_results set to true
        if plot_results:

            viz_utils.plot_postural_transitions(
                accel, self.gyro, accel_unit, gyro_unit, postural_transitions_, sampling_freq_Hz
            )

        # Return an instance of the class
        return self
    
    def spatio_temporal_parameters(self: Self) -> None:
        """
        Extracts spatio-temporal parameters of the detected postural transitions.

        Returns:
            The spatio-temporal parameter information is stored in the 'spatio_temporal_parameters'
            attribute, which is a pandas DataFrame as:
                - type_of_postural_transition: Type of postural transition which is either "sit to stand" or "stand to sit".
                - angel_of_postural_transition: Angle of the postural transition in degrees.
                - maximum_flexion_velocity: Maximum flexion velocity in deg/s.
                - maximum_extension_velocity: Maximum extension velocity in deg/s.
        """
        if self.postural_transitions_ is None:
            raise ValueError("No postural transition detected. Please run the detect method first.")

        # Initialize list for postural transition types
        postural_transition_type = []
        
        i = self.postural_transition_angle >= self.min_postural_transition_angle_deg
        postural_transition_type = [postural_transition_type[idx] for idx, val in enumerate(postural_transition_type) if i[idx]]

        # Distinguish between different types of postural transitions
        for i in range(len(self.local_peaks)):
            gyro_temp = self.gyro_mediolateral[self.left_side[i] : self.lef_side[i]]
            min_peak = np.min(gyro_temp)
            max_peak = np.max(gyro_temp)
            if (abs(min_peak) - max_peak) > 0.5:
                postural_transition_type.append("stand to sit")
            else:
                postural_transition_type.append("sit to stand")
        
        # Calculate postural transition angle
        valid_indices = self.postural_transition_angle >= 15
        self.postural_transition_angle = self.postural_transition_angle[valid_indices]

        # Calculate maximum flexion velocity and maximum extension velocity
        flexion_max_vel = np.zeros_like(self.local_peaks)
        extension_max_vel = np.zeros_like(self.local_peaks)

        for idx in range(len(self.local_peaks)):
            flexion_max_vel[idx] = max(abs(self.gyro_mediolateral[self.left_side[idx] : self.local_peaks[idx]]))
            extension_max_vel[idx] = max(abs(self.gyro_mediolateral[self.local_peaks[idx] : self.lef_side[idx]]))

        # Filter the velocities based on valid indices
        flexion_max_vel = [
            flexion_max_vel[idx] for idx, val in enumerate(flexion_max_vel) if valid_indices[idx]
        ]

        # Calculate maximum extension velocity
        extension_max_vel = [
            extension_max_vel[idx]
            for idx, val in enumerate(extension_max_vel)
            if valid_indices[idx]
        ]

        # Create a DataFrame with the calculated spatio-temporal parameters
        self.parameters_ = pd.DataFrame(
            {
                "type_of_postural_transition": postural_transition_type,
                "angle_of_postural_transition": self.postural_transition_angle,
                "maximum_flexion_velocity": flexion_max_vel,
                "maximum_extension_velocity": extension_max_vel,
            }
        )

        # Set the index name to 'postural transition id'
        self.parameters_.index.name = "postural transition id"