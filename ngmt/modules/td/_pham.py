# Import libraries
import numpy as np
import pandas as pd
import scipy
from ngmt.utils import preprocessing
from ngmt.utils import viz_utils
from typing import Optional, TypeVar

Self = TypeVar("Self", bound="PhamTurnDetection")

class PhamTurnDetection:
    """
    This algorithm aims to detect turns using accelerometer and gyroscope data collected from a lower back
    inertial measurement unit (IMU) sensor.

    The core of the algorithm lies in the detect method, where turns are identified using accelerometer and
    gyroscope data. The method first processes the gyro data, converting it to rad/s if needed and computing
    the variance to identify periods of low variance, which may indicate bias. It then calculates the gyro bias
    and subtracts it from the original gyro signal to remove any biases. Next, the yaw angle is computed by
    integrating the vertical component of the gyro data, and zero-crossings indices are found to detect turns.
    Then, turns are identified based on significant changes in the yaw angle.

    The algorithm also accounts for hesitations, which are brief pauses or fluctuations in the signal that may
    occur within a turn. Hesitations are marked based on specific conditions related to the magnitude and
    continuity of the yaw angle changes.

    Then, the detected turns are characterized by their onset and duration. Turns with angles equal to or greater
    than 90 degrees and durations between 0.5 and 10 seconds are selected for further analysis. Finally, the detected
    turns along with their characteristics (onset, duration, etc.) are stored in a pandas DataFrame
    (turns_ attribute).

    In addition, spatial-temporal parameters are calculated using detected turns and their characteristics by
    the spatio_temporal_parameters method. As a return, the turn id along with its spatial-temporal parameters
    including direction (left or right), angle of turn and peak angular velocity are stored in a pandas DataFrame
    (parameters_ attribute).

    Optionally, if plot_results is set to True, the algorithm generates a plot visualizing the accelerometer
    and gyroscope data alongside the detected turns. This visualization aids in the qualitative assessment of
    the algorithm's performance and provides insights into the dynamics of the detected turns.

    Methods:
        detect(data, gyro_vertical, accel_unit, gyro_unit, sampling_freq_Hz, dt_data, tracking_system, tracked_point, plot_results):
            Detects turns using accelerometer and gyro signals.

            Args:
                data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
                gyro_vertical (str): The column name that corresponds to the vertical component gyro.
                accel_unit (str): Unit of acceleration data.
                gyro_unit (str): Unit of gyro data.
                sampling_freq_Hz (float, int): Sampling frequency of the signals in Hz.
                dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
                tracking_system (str, optional): Tracking systems.
                tracked_point (str, optional): Tracked points on the body.
                plot_results (bool, optional): If True, generates a plot. Default is False.

            Returns:
                PhamTurnDetection: an instance of the class with the detected turns
                stored in the 'turns_' attribute.

        spatio_temporal_parameters():
            Extracts spatio-temporal parameters of the detected turns.

            Returns:
                pd.DataFrame: The spatio-temporal parameter information is stored in the 'spatio_temporal_parameters'
                attribute with the following columns:
                    - direction_of_turn: Direction of turn which is either "left" or "right".
                    - angle_of_turn: Angle of the turn in degrees.
                    - peak_angular_velocity: Peak angular velocity during turn in deg/s.

    Examples:
        >>> pham = PhamTurnDetection()
        >>> pham.detect(
                data=input_data,
                gyro_vertical="pelvis_GYRO_x",
                accel_unit="g",
                gyro_unit="rad/s",
                sampling_freq_Hz=200.0,
                tracking_system="imu",
                tracked_point="LowerBack",
                plot_results=False
                )
        >>> print(pham.turns_)
                onset   duration   event_type   tracking_systems    tracked_points
            0   4.04    3.26       turn         imu                 LowerBack
            1   9.44    3.35       turn         imu                 LowerBack

        >>> pham.spatio_temporal_parameters()
        >>> print(pham.parameters_)
                direction_of_turn   angle_of_turn   peak_angular_velocity
            0   left               -197.55          159.45
            1   right               199.69          144.67

    References:
        [1] Pham et al. (2017). Algorithm for Turning Detection and Analysis Validated under Home-Like Conditions...
    """

    def __init__(
        self,
        thr_gyro_var: float = 2e-4,
        min_turn_duration_s: float = 0.5,
        max_turn_duration_s: float = 10,
        min_turn_angle_deg: float = 90,
    ):
        """
        Initializes the PhamTurnDetection instance.

        Args:
            thr_gyro_var (float): Threshold value for identifying periods where the variance is low. Default is 2e-4.
            min_turn_duration_s (float): Minimum duration of a turn in seconds. Default is 0.5.
            max_turn_duration_s (float): Maximum duration of a turn in seconds. Default is 10.
            min_turn_angle_deg (float): Minimum angle of a turn in degrees. Default is 90.
        """
        self.thr_gyro_var = thr_gyro_var
        self.min_turn_duration_s = min_turn_duration_s
        self.max_turn_duration_s = max_turn_duration_s
        self.min_turn_angle_deg = min_turn_angle_deg

    def detect(
        self,
        data: pd.DataFrame,
        gyro_vertical: str,
        accel_unit: str,
        gyro_unit: str,
        sampling_freq_Hz: float,
        dt_data: Optional[pd.Series] = None,
        tracking_system: Optional[str] = None,
        tracked_point: Optional[str] = None,
        plot_results: bool = False,
    ) -> Self:
        """
        Detects truns based on the input accelerometer and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            gyro_vertical (str): The column name that corresponds to the vertical component gyro.
            accel_unit (str): Unit of acceleration data.
            gyro_unit (str): Unit of gyro data.
            sampling_freq_Hz (float): Sampling frequency of the input data in Hz.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
            tracking_system (str, optional): Tracking systems.
            tracked_point (str, optional): Tracked points on the body.
            plot_results (bool, optional): If True, generates a plot. Default is False.

        Returns:
            The turns information is stored in the 'turns_' attribute,
            which is a pandas DataFrame in BIDS format with the following information:
                - onset: Start time of the turn in second.
                - duration: Duration of the turn in second.
                - event_type: Type of the event (turn).
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

        # Check if data has the correct shape
        if data.shape[1] != 6:
            raise ValueError(
                "Input data must have 6 columns (3 for accelerometer and 3 for gyro)"
            )

        # Check if sampling frequency is positive
        if sampling_freq_Hz <= 0:
            raise ValueError("Sampling frequency must be positive")

        # Check if plot_results is a boolean
        if not isinstance(plot_results, bool):
            raise ValueError("plot_results must be a boolean value")

        # Identify the columns in the DataFrame that correspond to accelerometer data
        accel_columns = [col for col in data.columns if 'ACCEL' in col]
        
        # Identify the columns in the DataFrame that correspond to gyroscope data
        gyro_columns = [col for col in data.columns if 'GYRO' in col]

        # Ensure that there are exactly 3 columns each for accelerometer and gyroscope data
        if len(accel_columns) != 3 or len(gyro_columns) != 3:
            raise ValueError("Data must contain 3 accelerometer and 3 gyroscope columns.")

        # Select acceleration data and convert it to numpy array format
        accel = data[accel_columns].copy().to_numpy()

        # Select gyro data and convert it to numpy array format
        gyro = data[gyro_columns].copy().to_numpy()
        self.gyro = gyro
        
        # Check unit of acceleration data if it is in g or m/s^2 (including variations)
        if accel_unit in ["m/s^2", "meters/s^2", "meter/s^2"]:
            # Convert acceleration data from m/s^2 to g (if not already is in g)
            accel /= 9.81
        elif accel_unit in ["g", "G"]:
            pass  # No conversion needed
        else:
            raise ValueError("Unsupported unit for acceleration data")

        # Check unit of gyro data if it is in deg/s or rad/s
        if gyro_unit in ["deg/s", "degrees per second"]:
            # Convert gyro data from deg/s to rad/s (if not already is in rad/s)
            gyro = np.deg2rad(gyro)
        elif gyro_unit in ["rad/s", "radians per second"]:
            pass  # Gyro data is already in rad/s
        else:
            raise ValueError("Invalid unit for gyro data. Must be 'deg/s' or 'rad/s'")

        # Compute the variance of the moving window of gyro signal
        gyro_vars = []

        for i in range(3):
            gyro_var = preprocessing.moving_var(gyro[:, i], sampling_freq_Hz)
            gyro_vars.append(gyro_var)
        gyro_vars = np.array(gyro_vars)
        gyro_vars = np.mean(gyro_vars, axis=0)

        # Find bias period
        bias_periods = np.where(gyro_vars < self.thr_gyro_var)[0]

        # Remove the last sampling_freq_Hz samples from bias_periods to avoid edge effects
        bias_periods = bias_periods[bias_periods < (len(gyro_vars) - sampling_freq_Hz)]

        # Compute gyro bias (mean of gyro signal during bias periods)
        self.gyro_bias = np.mean(gyro[bias_periods, :], axis=0)

        # Subtract gyro bias from the original gyro signal
        gyro_unbiased = gyro - self.gyro_bias

        # Get the index of the vertical component of gyro from data
        gyro_vertical_index = [i for i, col in enumerate(gyro_columns) if gyro_vertical in col][0]

        # Integrate x component of the gyro signal to get yaw angle (also convert gyro unit to deg/s)
        self.yaw = (
            scipy.integrate.cumtrapz(
                np.rad2deg(gyro_unbiased[:, gyro_vertical_index]), initial=0
            )
            / sampling_freq_Hz
        )

        # Find zero-crossings indices
        self.index_zero_crossings = np.where(
            np.diff(np.sign(gyro[:, gyro_vertical_index]))
        )[0]

        # Calculate turns from yaw angle
        self.turns_all = (
            self.yaw[self.index_zero_crossings[1:]]
            - self.yaw[self.index_zero_crossings[:-1]]
        )

        # Marks hesitations in the signal
        # Initialize an array to mark hesitations
        hesitation_markers = np.zeros(len(self.turns_all))

        # Loop through each index in the turns_all array
        for i in range(len(self.turns_all)):
            # Check if the absolute value of the turn angle at index i is greater than or equal to 10
            if abs(self.turns_all[i]) >= 10:
                # Loop to search for potential hesitations
                for j in range(i + 1, len(self.turns_all)):
                    # Check if the difference between current index and i exceeds 4, or if the time between zero crossings exceeds half a second
                    if (j - i) > 4 or (
                        self.index_zero_crossings[j] - self.index_zero_crossings[i + 1]
                        > (sampling_freq_Hz / 2)
                    ):
                        # Break the loop if the conditions for hesitation are not met
                        break
                    else:
                        # Check conditions for hesitation:
                        # - Absolute values of both turns are greater than or equal to 10
                        # - The relative change in yaw angle is less than 20% of the minimum turn angle
                        # - The signs of both turns are the same
                        if (
                            abs(self.turns_all[i]) >= 10
                            and abs(self.turns_all[j]) >= 10
                            and abs(
                                self.yaw[self.index_zero_crossings[i + 1]]
                                - self.yaw[self.index_zero_crossings[j]]
                            )
                            / min(abs(self.turns_all[i]), abs(self.turns_all[j]))
                            < 0.2
                            and np.sign(self.turns_all[i]) == np.sign(self.turns_all[j])
                        ):
                            # Mark the range between i and j (inclusive) as a hesitation
                            hesitation_markers[i : j + 1] = 1
                            # Break the inner loop as the hesitation condition is met
                            break

        # Initialize variables to store data related to turns without hesitation
        sum_temp = 0  # Temporary sum for accumulating turn angles
        turns_no_hesitation = []  # List to store turn angles without hesitation
        flags_start_no_hesitation = (
            []
        )  # List to store start indices of turns without hesitation
        flags_end_no_hesitation = (
            []
        )  # List to store end indices of turns without hesitation
        durations_no_hesitation = (
            []
        )  # List to store durations of turns without hesitation
        z = 1  # Index for keeping track of the current turn

        # Iterate through each index in the hesitation_markers array
        for i in range(len(hesitation_markers)):
            # Check if there is a hesitation marker at the current index
            if hesitation_markers[i] == 1:
                # Check if sum_temp is zero, indicating the start of a new turn
                if sum_temp == 0:
                    f1 = self.index_zero_crossings[
                        i
                    ]  # Store the start index of the turn

                # Check if the absolute value of the turn angle is greater than or equal to 10
                if abs(self.turns_all[i]) >= 10:
                    try:
                        # Check if the next index also has a hesitation marker
                        if hesitation_markers[i + 1] != 0:
                            # Iterate through subsequent indices to find the end of the turn
                            for j in range(i + 1, len(hesitation_markers)):
                                # Check if the absolute value of the turn angle is greater than or equal to 10
                                if abs(self.turns_all[j]) >= 10:
                                    # Check if the signs of the turn angles are the same
                                    if np.sign(self.turns_all[j]) == np.sign(
                                        self.turns_all[i]
                                    ):
                                        sum_temp += self.turns_all[
                                            i
                                        ]  # Accumulate the turn angle
                                    else:
                                        f2 = hesitation_markers[
                                            i + 1
                                        ]  # Store the end index of the turn
                                        sum_temp += self.turns_all[
                                            i
                                        ]  # Accumulate the turn angle
                                        turns_no_hesitation.append(
                                            sum_temp
                                        )  # Store the turn angle without hesitation
                                        flags_start_no_hesitation.append(
                                            f1
                                        )  # Store the start index of the turn
                                        flags_end_no_hesitation.append(
                                            f2
                                        )  # Store the end index of the turn
                                        durations_no_hesitation.append(
                                            (f2 - f1) / sampling_freq_Hz
                                        )  # Calculate and store the duration of the turn
                                        z += 1  # Increment the turn index
                                        sum_temp = 0  # Reset the temporary sum
                                        del (
                                            f1,
                                            f2,
                                        )  # Delete stored indices to avoid conflicts
                                    break  # Exit the loop once the turn is processed
                        else:
                            f2 = self.index_zero_crossings[
                                i + 1
                            ]  # Store the end index of the turn
                            sum_temp += self.turns_all[i]  # Accumulate the turn angle
                            turns_no_hesitation.append(
                                sum_temp
                            )  # Store the turn angle without hesitation
                            flags_start_no_hesitation.append(
                                f1
                            )  # Store the start index of the turn
                            flags_end_no_hesitation.append(
                                f2
                            )  # Store the end index of the turn
                            durations_no_hesitation.append(
                                (f2 - f1) / sampling_freq_Hz
                            )  # Calculate and store the duration of the turn
                            z += 1  # Increment the turn index
                            del f1, f2  # Delete stored indices to avoid conflicts
                            sum_temp = 0  # Reset the temporary sum
                    except:
                        f2 = self.index_zero_crossings[
                            i + 1
                        ]  # Store the end index of the turn
                        sum_temp += self.turns_all[i]  # Accumulate the turn angle
                        turns_no_hesitation.append(
                            sum_temp
                        )  # Store the turn angle without hesitation
                        flags_start_no_hesitation.append(
                            f1
                        )  # Store the start index of the turn
                        flags_end_no_hesitation.append(
                            f2
                        )  # Store the end index of the turn
                        durations_no_hesitation.append(
                            (f2 - f1) / sampling_freq_Hz
                        )  # Calculate and store the duration of the turn
                        z += 1
                        del f1, f2
                        sum_temp = 0
                else:
                    sum_temp += self.turns_all[
                        i
                    ]  # Accumulate the turn angle if it's smaller than 10 degrees
            else:  # If there's no hesitation marker at the current index
                turns_no_hesitation.append(self.turns_all[i])
                flags_start_no_hesitation.append(self.index_zero_crossings[i])
                flags_end_no_hesitation.append(self.index_zero_crossings[i + 1])
                durations_no_hesitation.append(
                    (self.index_zero_crossings[i + 1] - self.index_zero_crossings[i])
                    / sampling_freq_Hz
                )  # Calculate and store the duration of the turn
                z += 1  # Increment the turn index

        # Initialize lists to store information about turns >= 90 degrees
        turns_90 = []
        flags_start_90 = []
        flags_end_90 = []

        # Iterate through each turn without hesitation
        for k in range(len(turns_no_hesitation)):
            # Check if the turn angle is greater than or equal to 90 degrees
            # and if the duration of the turn is between 0.5 and 10 seconds
            if (
                abs(turns_no_hesitation[k]) >= self.min_turn_angle_deg
                and durations_no_hesitation[k] >= self.min_turn_duration_s
                and durations_no_hesitation[k] < self.max_turn_duration_s
            ):
                # If conditions are met, store information about the turn
                turns_90.append(turns_no_hesitation[k])
                flags_start_90.append(flags_start_no_hesitation[k])
                flags_end_90.append(flags_end_no_hesitation[k])

        # Initialize lists to store additional information about >= 90 degree turns
        duration_90 = []

        # Assign detected truns attribute
        self.turns_90 = turns_90
        self.flags_start_90 = flags_start_90
        self.flags_end_90 = flags_end_90

        # Assign sampling frequency to the attribute
        self.sampling_freq_Hz = sampling_freq_Hz

        # Compute duration of the turn in seconds
        for k in range(len(flags_start_90)):
            # Compute duration of the turn in seconds
            duration_nsamples = self.flags_end_90[k] - self.flags_start_90[k]
            duration_90.append(duration_nsamples / sampling_freq_Hz)

        # Create a DataFrame with postural transition information
        self.turns_ = pd.DataFrame(
            {
                "onset": np.array(flags_start_90) / sampling_freq_Hz,
                "duration": duration_90,
                "event_type": "turn",
                "tracking_systems": tracking_system,
                "tracked_points": tracked_point,
            }
        )

        # If original datetime is available, update the 'onset' and 'duration'
        if dt_data is not None:
            # Update the 'onset' based on the original datetime information
            self.turns_["onset"] = dt_data.iloc[flags_start_90].reset_index(drop=True)

            # Update the 'duration' based on the difference between end and start indices
            self.turns_["duration"] = dt_data.iloc[flags_end_90].reset_index(
                drop=True
            ) - dt_data.iloc[flags_start_90].reset_index(drop=True)

        # If Plot_results set to true
        if plot_results:
            viz_utils.plot_turns(
                accel, gyro, accel_unit, gyro_unit, self.turns_, sampling_freq_Hz
            )

        # Return an instance of the class
        return self

    def spatio_temporal_parameters(self: Self) -> None:
        """
        Extracts spatio-temporal parameters of the detected turns.

        Returns:
            The spatio-temporal parameter information is stored in the 'spatio_temporal_parameters'
            attribute, which is a pandas DataFrame as:
                - direction_of_turn: Direction of turn which is either "left" or "right".
                - angle_of_turn: Angle of the turn in degrees.
                - peak_angular_velocity: Peak angular velocity during turn in deg/s.
        """
        if self.turns_ is None:
            raise ValueError("No turns detected. Please run the detect method first.")

        # Calculate additional information for each >= 90 degree turn
        peak_angular_velocities = []
        diff_yaw = np.diff(self.yaw)

        for k in range(len(self.flags_start_90)):
            # Calculate peak angular velocity during the turn
            diff_vector = abs(
                diff_yaw[(self.flags_start_90[k] - 1) : (self.flags_end_90[k] - 1)]
            )
            peak_angular_velocities.append(np.max(diff_vector) * self.sampling_freq_Hz)

        # Determine direction of the turn (left or right)
        direction_of_turns = []

        for turn_angle in self.turns_90:
            if turn_angle < 0:
                direction_of_turns.append("left")
            else:
                direction_of_turns.append("right")

        # Create a DataFrame with the calculated spatio-temporal parameters
        self.parameters_ = pd.DataFrame(
            {
                "direction_of_turn": direction_of_turns,
                "angle_of_turn": self.turns_90,
                "peak_angular_velocity": peak_angular_velocities,
            }
        )

        # Set the index name to 'turn id'
        self.parameters_.index.name = "turn id"