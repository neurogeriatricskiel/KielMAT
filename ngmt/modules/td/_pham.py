# Import libraries
import numpy as np
import pandas as pd
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


class PhamTurnDetection:
    """
    This algorithm aims to detect turns using accelerometer and gyroscope data collected from a lower back 
    inertial measurement unit (IMU) sensor.

    Describe algorithm functionality here...

    Attributes:
        add attributes here....
        tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
        tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.

    Methods:
        detect(data, sampling_freq_Hz):
            Detects turns using accelerometer and gyro signals.

            Args:
                data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
                sampling_freq_Hz (float, int): Sampling frequency of the signals.
                plot_results (bool, optional): If True, generates a plot. Default is False.

            Returns:
                PhamTurnDetection: an instance of the class with the detected turns
                stored in the 'detected_turns' attribute.

    Examples:
        >>> pham = PhamTurnDetection()
        >>> pham.detect(
                data=input_data,
                sampling_freq_Hz=,
                )
        >>> print(pham.detected_turns)
                onset      duration    event_type       tracking_systems    tracked_points
            0   17.895     1.800       turn             imu                 LowerBack
            1   54.655     1.905       turn             imu                 LowerBack

    References:
        [1] Pham et al. (2017). Algorithm for Turning Detection and Analysis Validated under Home-Like Conditions...
    """

    def __init__(
        self,
        tracking_systems: str = "imu",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the PhamTurnDetection instance.

        Args:
            tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.detected_turns = None

    def detect(
        self, data: pd.DataFrame, sampling_freq_Hz: float, plot_results: bool = False
    ) -> pd.DataFrame:
        """
        Detects truns based on the input accelerometer and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the input data.
            plot_results (bool, optional): If True, generates a plot. Default is False.

        Returns:
            The turns information is stored in the 'detected_turns' attribute,
            which is a pandas DataFrame in BIDS format.
        """
        # Assign the DataFrame to the 'detected_turns' attribute

        #self.detected_turns = 

        # Return an instance of the class
        return self
