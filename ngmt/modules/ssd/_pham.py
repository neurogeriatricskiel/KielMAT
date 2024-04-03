# Import libraries
import numpy as np
import pandas as pd
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


class PhamSittoStandStandtoSitDetection:
    """
    This algorithm aims to detect postural transitions (e.g., sit-to-stand or stand-to-sit movements)
    using accelerometer and gyroscope data collected from a lower back inertial measurement unit (IMU)
    sensor.

    The algorithm is designed to be robust in detecting sit-to-stand and stand-to-sit transitions
    using inertial sensor data and provides detailed information about these transitions. It starts by
    loading the accelerometer and gyro data, which includes three columns corresponding to the acceleration
    and gyro signals across the x, y, and z axes, along with the sampling frequency of the data. It first
    checks the validity of the input data. Then, it calculates the sampling period, selects accelerometer
    and gyro data. Tilt angle estimation is performed using gyro data. The tilt angle is decomposed using
    wavelet transformation to identify stationary periods. Stationary periods are detected using accelerometer
    variance and gyro variance.Then, peaks in the wavelet-transformed tilt signal are detected as potential
    postural transition events.

    If there's enough stationary data, further processing is done to estimate the orientation using
    quaternions and to identify the beginning and end of postural transitions using gyro data. Otherwise,
    if there's insufficient stationary data, direction changes in gyro data are used to infer postural
    transitions.

    Finally, the detected postural transitions are classified as either sit-to-stand or stand-to-sit
    based on gyro data characteristics and other criteria. The detected postural transitions along with
    their characteristics (onset time, duration, event type, angle, maximum flexion/extension velocity,
    tracking systems, and tracked points) are stored in a pandas DataFrame (postural_transitions_ attribute).

    If requested (plot_results set to True), it generates plots of the accelerometer and gyroscope data
    along with the detected postural transitions.

    Attributes:
        cutoff_freq_hz (float, optional): Cutoff frequency for low-pass Butterworth filer. Default is 5.
        accel_convert_unit (float): Conversion of acceleration unit from g to m/s^2
        tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
        tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.

    Methods:
        detect(data, sampling_freq_Hz):
            Detects  sit to stand and stand to sit events using accelerometer and gyro signals.

    Examples:
        >>> pham = PhamSittoStandStandtoSitDetection()
        >>> pham.detect(
                data=df_data,
                sampling_freq_Hz=200,
                )
        >>> print(pham.postural_transitions_)
                onset      duration    event_type       postural transition angle [°]   maximum flexion velocity [°/s]  maximum extension velocity [°/s]  tracking_systems    tracked_points
            0   17.895     1.800       sit to stand     53.263562                       79                              8                                 imu                 LowerBack
            1   54.655     1.905       stand to sit     47.120448                       91                              120                               imu                 LowerBack
            2   56.020     1.090       sit to stand     23.524748                       62                              10                                imu                 LowerBack
            3   135.895    2.505       stand to sit     21.764146                       40                              65                                imu                 LowerBack

    References:
        [1] Pham et al. (2018). Validation of a Lower Back "Wearable"-Based Sit-to-Stand and Stand-to-Sit Algorithm...
    """

    def __init__(
        self,
        cutoff_freq_hz: float = 5.0,
        accel_convert_unit=9.81,
        tracking_systems: str = "imu",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the PhamSittoStandStandtoSitDetection instance.

        Args:
            cutoff_freq_hz (float, optional): Cutoff frequency for low-pass Butterworth filer. Default is 5.
            accel_convert_unit (float): Conevrsion of acceleration unit from g to m/s^2
            tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.cutoff_freq_hz = cutoff_freq_hz
        self.accel_convert_unit = accel_convert_unit
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.postural_transitions_ = None

    def detect(
        self, data: pd.DataFrame, sampling_freq_Hz: float, plot_results: bool = False, dt_data: pd.Series = None
    ) -> pd.DataFrame:
        """
        Detects sit to stand and stand to sit based on the input acceleromete and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the input data.
            plot_results (bool, optional): If True, generates a plot. Default is False.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.

            Returns:
                PhamSittoStandStandtoSitDetection: Returns an instance of the class.
                    The postural transition information is stored in the 'postural_transitions_' attribute,
                    which is a pandas DataFrame in BIDS format with the following columns:
                        - onset: Start time of the postural transition [s].
                        - duration: Duration of the postural transition [s].
                        - event_type: Type of the event (sit to stand ot stand to sit).
                        - postural transition angle: Angle of the postural transition in degree [°].
                        - maximum flexion velocity: Maximum flexion velocity [°/s].
                        - maximum extension velocity: Maximum extension velocity [°/s].
                        - tracking_systems: Tracking systems used (default is 'imu').
                        - tracked_points: Tracked points on the body (default is 'LowerBack').
        """        
        # Error handling for invalid input data
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 6:
            raise ValueError(
                "Input accelerometer and gyro data must be a DataFrame with 6 columns for x, y, and z axes."
            )

        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (
            not isinstance(dt_data, pd.Series)
            or not pd.api.types.is_datetime64_any_dtype(dt_data)
        ):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Calculate sampling period
        sampling_period = 1 / sampling_freq_Hz

        # Select acceleration data and convert it to numpy array format
        accel = data.iloc[:, 0:3].copy()
        accel = accel.to_numpy()

        # Select gyro data and convert it to numpy array format
        gyro = data.iloc[:, 3:6].copy()
        gyro = gyro.to_numpy()

        # Calculate timestamps to use in next calculation
        time = np.arange(1, len(accel[:, 0]) + 1) * sampling_period

        # Estimate tilt angle in deg
        tilt_angle_deg = preprocessing.tilt_angle_estimation(
            data=gyro, sampling_frequency_hz=sampling_freq_Hz
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

        # Find peaks in denoised tilt signal:  peaks of the tilt_dwt signal with magnitude and prominence >0.1 were defined as PT events
        local_peaks, _ = scipy.signal.find_peaks(tilt_dwt, height=0.2, prominence=0.2)

        # Calculate the norm of acceleration
        acceleration_norm = np.sqrt(
            accel[:, 0] ** 2 + accel[:, 1] ** 2 + accel[:, 2] ** 2
        )

        # Calculate absolute value of the acceleration signal
        acceleration_norm = np.abs(acceleration_norm)

        # Detect stationary parts of the signal based on the deifned threshold
        stationary_1 = acceleration_norm < 0.05
        stationary_1 = (stationary_1).astype(int)

        # Compute the variance of the moving window acceleration
        accel_var = preprocessing.moving_var(
            data=acceleration_norm, window=sampling_freq_Hz
        )

        # Calculate stationary_2 from acceleration variance
        threshold = 10**-2
        stationary_2 = accel_var <= threshold
        stationary_2 = (accel_var <= threshold).astype(int)

        # Calculate stationary of gyro variance
        gyro_norm = np.sqrt(gyro[:, 0] ** 2 + gyro[:, 1] ** 2 + gyro[:, 2] ** 2)

        # Compute the variance of the moving window of gyro
        gyro_var = preprocessing.moving_var(data=gyro_norm, window=sampling_freq_Hz)

        # Calculate stationary of gyro variance
        stationary_3 = gyro_var <= threshold
        stationary_3 = (gyro_var <= threshold).astype(int)

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
            time_pt, pt_type, pt_angle, duration, flexion_max_vel, extension_max_vel = (
                preprocessing.process_postural_transitions_stationary_periods(
                    time,
                    accel,
                    gyro,
                    stationary,
                    tilt_angle_deg,
                    sampling_period,
                    sampling_freq_Hz,
                    init_period,
                    local_peaks,
                )
            )

            # Create a DataFrame with postural transition information
            postural_transitions_ = pd.DataFrame(
                {
                    "onset": time_pt,
                    "duration": duration,
                    "event_type": pt_type,
                    "postural transition angle": pt_angle,
                    "maximum flexion velocity": flexion_max_vel,
                    "maximum extension velocity": extension_max_vel,
                    "tracking_systems": self.tracking_systems,
                    "tracked_points": self.tracked_points,
                }
            )

        else:
            # Handle cases where there is not enough stationary data
            # Find indices where the product of consecutive changes sign, indicating a change in direction
            iZeroCr = np.where((gyro[:, 1][:-1] * gyro[:, 1][1:]) < 0)[0]

            # Calculate the difference between consecutive values
            gyrY_diff = np.diff(gyro[:, 1])

            # Initialize arrays to store left and right indices for each local peak
            # Initialize left side indices with ones
            ls = np.ones_like(local_peaks)

            # Initialize right side indices with length of gyro data
            rs = len(gyro[:, 1]) * np.ones_like(local_peaks)

            # Loop through each local peak
            for i in range(len(local_peaks)):
                # Get the index of the current local peak
                pt = local_peaks[i]

                # Calculate distances to all zero-crossing points relative to the peak
                dist2peak = iZeroCr - pt

                # Extract distances to zero-crossing points on the left side of the peak
                dist2peak_ls = dist2peak[dist2peak < 0]

                # Extract distances to zero-crossing points on the right side of the peak
                dist2peak_rs = dist2peak[dist2peak > 0]

                # Iterate over distances to zero-crossing points on the left side of the peak (in reverse order)
                for j in range(len(dist2peak_ls) - 1, -1, -1):
                    # Check if slope is down and the left side not too close to the peak (more than 200ms)
                    if gyrY_diff[pt + dist2peak_ls[j]] < 0 and -dist2peak_ls[j] > 25:
                        if j > 0:  # Make sure dist2peak_ls[j] exist
                            # If the left side peak is far enough or small enough
                            if (dist2peak_ls[j] - dist2peak_ls[j - 1]) >= 25 or (
                                tilt_angle_deg[pt + dist2peak_ls[j - 1]]
                                - tilt_angle_deg[pt + dist2peak_ls[j]]
                            ) > 1:
                                # Store the index of the left side
                                ls[i] = pt + dist2peak_ls[j]
                                break
                        else:
                            ls[i] = pt + dist2peak_ls[j]
                            break
                for j in range(len(dist2peak_rs)):
                    if gyrY_diff[pt + dist2peak_rs[j]] < 0 and dist2peak_rs[j] > 25:
                        rs[i] = pt + dist2peak_rs[j]
                        break

            # Initialize list for PT types
            pt_type = []

            # Distinguish between different types of postural transitions
            for i in range(len(local_peaks)):
                gyro_temp = gyro[:, 1][ls[i] : rs[i]]
                min_peak = np.min(gyro_temp)
                max_peak = np.max(gyro_temp)
                if (abs(min_peak) - max_peak) > 0.5:
                    pt_type.append("sit to stand")
                else:
                    pt_type.append("stand to sit")

            # Calculate maximum flexion velocity and maximum extension velocity
            flexion_max_vel = np.zeros_like(local_peaks)
            extension_max_vel = np.zeros_like(local_peaks)
            for i in range(len(local_peaks)):
                flexion_max_vel[i] = max(abs(gyro[:, 1][ls[i] : local_peaks[i]]))
                extension_max_vel[i] = max(abs(gyro[:, 1][local_peaks[i] : rs[i]]))

            # Calculate PT angle
            pt_angle = np.abs(tilt_angle_deg[local_peaks] - tilt_angle_deg[ls])
            if ls[0] == 1:
                pt_angle[0] = abs(
                    tilt_angle_deg[local_peaks[0]] - tilt_angle_deg[rs[0]]
                )

            # Calculate duration of each PT
            duration = (rs - ls) / sampling_freq_Hz

            # Convert peak times to integers
            time_pt = time[local_peaks]

        # Remove too small postural transitions
        i = pt_angle >= 15
        time_pt = time_pt[i]
        pt_type = [pt_type[idx] for idx, val in enumerate(pt_type) if i[idx]]
        pt_angle = pt_angle[i]
        duration = duration[i]
        flexion_max_vel = [
            flexion_max_vel[idx] for idx, val in enumerate(flexion_max_vel) if i[idx]
        ]
        extension_max_vel = [
            extension_max_vel[idx]
            for idx, val in enumerate(extension_max_vel)
            if i[idx]
        ]

        # Check if dt_data is provided for datetime conversion
        if dt_data is not None:
            # Convert onset times to datetime format
            starting_datetime = dt_data.iloc[0]  # Assuming dt_data is aligned with the signal data
            time_pt = [starting_datetime + pd.Timedelta(seconds=t) for t in time_pt]


        # Create a DataFrame with postural transition information
        postural_transitions_ = pd.DataFrame(
            {
                "onset": time_pt,
                "duration": duration,
                "event_type": pt_type,
                "postural transition angle": pt_angle,
                "maximum flexion velocity": flexion_max_vel,
                "maximum extension velocity": extension_max_vel,
                "tracking_systems": self.tracking_systems,
                "tracked_points": self.tracked_points,
            }
        )

        # Assign the DataFrame to the 'postural_transitions_' attribute
        self.postural_transitions_ = postural_transitions_

        # If Plot_results set to true
        # currently no plotting for datetime values
        if dt_data is not None and plot_results:
            print("No plotting for datetime values.")
            plot_results = False
            return self
        
        # If Plot_results set to true
        if plot_results:

            preprocessing.pham_plot_results(
                accel, gyro, postural_transitions_, sampling_freq_Hz
            )

        # Return an instance of the class
        return self
