# Import libraries
import numpy as np
import pandas as pd
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


class PhamSittoStandStandtoSitDetection:
    """
    Detects postural transitions (sit to stand and stand to sit) using acceleromter and gyro data.

    Attributes:
        event_type (str): Type of the detected event. Default is 'sit to satnd and stand to sit'.
        tracking_systems (str): Tracking systems used. Default is 'imu'.
        tracked_points (str): Tracked points on the body. Default is 'LowerBack'.
        postural_transitions_ (pd.DataFrame): DataFrame containing sit to stand and stand to sit information in BIDS format.

    Description:
        The algorithm finds sit to stand and stand to sit using acceleration and gyro signals from lower back IMU sensor.

        The accelerometer and gyroscope data are preprocessed and the stationary periods of 
        the lower back are identified. 
        
        These stationary periods are used to estimate the tilt angle with respect to the horizontal plane. 
        They were smoothed by using discrete wavelet transformation and the start and end of 
        the postural transitions are identified. 

        The sensor orientation during the postural transitions are estimated using the quaternion 
        (estimated from accelerometer and gyroscope data) to estimate the vertical displacement of the lower back. 

        Based on the extent of vertical displacement, postural transitions were classified as “effective postural transitions
        and “postural transitions attempts,” and the direction of the postural transitions was defined.

    Methods:
        detect(data, sampling_freq_Hz):
            Detects  sit to stand and stand to sit using accelerometer and gyro signals.

        __init__():
            Initializes the sit to stand and stand to sit instance.

            Args:
                data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
                sampling_freq_Hz (float, int): Sampling frequency of the signals.

            Returns:
                PhamSittoStandStandtoSitDetection: Returns an instance of the class.
                    The postural transition information is stored in the 'postural_transitions' attribute,
                    which is a pandas DataFrame in BIDS format with the following columns:
                        - onset: Start time of the postural transition.
                        - duration: Duration of the postural transition.
                        - event_type: Type of the event (sit to stand or stand to sit).
                        - tracking_systems: Tracking systems used (default is 'imu').
                        - tracked_points: Tracked points on the body (default is 'LowerBack').

        Examples:
            Determines sit to stand and stand to sit in the sensor signal.

            >>> pham = PhamSittoStandStandtoSitDetection()
            >>> pham.detect(
                    data=,
                    sampling_freq_Hz=100,
                    )
            >>> postural_transitions = pham.postural_transitions_
            >>> print(postural_transitions)
                    onset   duration    event_type      tracking_systems    tracked_points
                0   4.500   5.25        sit to satnd    imu                 LowerBack
                1   90.225  10.30       stand to sit    imu                 LowerBack

        References:
            [1] Pham et al. (2018). Validation of a Lower Back "Wearable"-Based Sit-to-Stand and 
            Stand-to-Sit Algorithm for Patients With Parkinson's Disease and Older Adults in a Home-Like 
            Environment. Frontiers in Neurology, 9, 652. https://doi.org/10.3389/fneur.2018.00652

    """

    def __init__(
        self,
        cutoff_freq_hz: float = 5.0,
        tracking_systems: str = "imu",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the PhamSittoStandStandtoSitDetection instance.

        Args:
            cutoff_freq_hz (float, optional): Cutoff frequency for low-pass Butterworth filer. Default is 5.
            event_type (str, optional): Type of the detected event. Default is ''.
            tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.cutoff_freq_hz = cutoff_freq_hz
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.stand_to_sit_sit_to_stand_ = None

    def detect(
        self, data: pd.DataFrame, sampling_freq_Hz: float
    ) -> pd.DataFrame:
        """
        Detects sit to stand and stand to sit based on the input acceleromete and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the input data.

            Returns:
                PhamSittoStandStandtoSitDetection: Returns an instance of the class.
                    The postural transition information is stored in the 'stand_to_sit_sit_to_stand' attribute,
                    which is a pandas DataFrame in BIDS format with the following columns:
                        - onset: Start time of the postural transition.
                        - duration: Duration of the postural transition.
                        - event_type: Type of the event (sit to stand ot stand to sit).
                        - tracking_systems: Tracking systems used (default is 'imu').
                        - tracked_points: Tracked points on the body (default is 'LowerBack').
        """
        # Error handling for invalid input data
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 6:
            raise ValueError(
                "Input accelerometer and gyro data must be a DataFrame with 6 columns for x, y, and z axes."
            )

        # Calculate sampling period
        sampling_period = 1/sampling_freq_Hz

        # Select acceleration data and convert it to numpy array format 
        accel = data.iloc[:, 0:3].copy()
        accel = accel.to_numpy()

        # Select gyro data and convert it to numpy array format 
        gyro = data.iloc[:, 3:6].copy()
        gyro = gyro.to_numpy()

        # Calculate timestamps to use in next calculation
        time = np.arange(1, len(accel[:, 0] ) + 1) * sampling_period

        # Estimate tilt angle in deg
        tilt_angle_deg = preprocessing.tilt_angle_estimation(data = gyro, sampling_frequency_hz = sampling_freq_Hz) 
        
        # Convert tilt angle to rad
        tilt_angle_rad = tilt_angle_deg * np.pi / 180

        # Calculate sine of the tilt angle in radians
        tilt_sin = np.sin(tilt_angle_rad)

        # Apply wavelet decomposition with level of 3
        tilt_denoise_3 = preprocessing.wavelet_decomposition(data = tilt_sin, level = 3, wavetype='coif5')

        # Apply wavelet decomposition with level of 10
        tilt_denoise_10 = preprocessing.wavelet_decomposition(data = tilt_sin, level = 10, wavetype='coif5')

        # Calculate difference 
        tilt_denoise = tilt_denoise_3 - tilt_denoise_10

        # Find peaks in denoised tilt signal
        local_peaks, _ = scipy.signal.find_peaks(tilt_denoise, height=0.2, prominence=0.2)

        # Calculate the norm of acceleration
        acceleration_norm = np.sqrt(accel[:,0] ** 2 + accel[:,1] ** 2 + accel[:,2] ** 2)

        # Applying highpass filter to the norm of the acceleartion data
        accel_norm_filtered = preprocessing.highpass_filtering(signal=acceleration_norm, order=1, method="butter", cutoff_freq_hz=0.001, sampling_freq_hz = sampling_freq_Hz)

        # Calculate absolute value of the acceleration signal
        accel_norm_filtered = np.abs(accel_norm_filtered)

        # Applying lowpass filter to the signal
        accel_norm_filtered = preprocessing.lowpass_filter(accel_norm_filtered, method="butter", order=1, cutoff_freq_hz=self.cutoff_freq_hz, sampling_rate_hz=sampling_freq_Hz)
       
        # Detect stationary parts of the signal based on the deifned threshold
        stationary_1 = accel_norm_filtered < 0.05
        stationary_1 = (stationary_1).astype(int)

        # Compute the variance of the moving window acceleration
        accel_var = preprocessing.moving_var(data = acceleration_norm, window = sampling_freq_Hz)
        
        # Calculate stationary_2 from acceleration variance
        threshold = 10**-2
        stationary_2 = accel_var <= threshold
        stationary_2 = (accel_var <= threshold).astype(int)

        # Calculate stationary of gyro variance
        gyro_norm = np.sqrt(gyro[:,0] ** 2 + gyro[:,1] ** 2 + gyro[:,2] ** 2)

        # Compute the variance of the moving window of gyro
        gyro_var = preprocessing.moving_var(data = gyro_norm, window = sampling_freq_Hz)

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
        
        # Assign the DataFrame to the 'postural_transitions_' attribute
        #self.postural_transitions_ = postural_transitions_

        # Return an instance of the class
        return self

