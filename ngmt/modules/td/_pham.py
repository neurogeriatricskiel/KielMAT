# Import libraries
import numpy as np
import pandas as pd
import scipy
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors 


class PhamTurnDetection:
    """
    This algorithm aims to detect turns using accelerometer and gyroscope data collected from a lower back 
    inertial measurement unit (IMU) sensor.

    The core of the algorithm lies in the detect method, where turns are identified using accelerometer and 
    gyroscope data. The method first processes the gyro data, converting it to rad/s and computing 
    the variance to identify periods of low variance, which may indicate bias. It then calculates the gyro bias 
    and subtracts it from the original gyro signal to remove any biases. Next, the yaw angle is computed by 
    integrating the gyro data, and zero-crossings indices are found to detect turns. Then, turns are identified 
    based on significant changes in the yaw angle. 
    
    The algorithm also accounts for hesitations, which are brief pauses or fluctuations in the signal that may 
    occur within a turn. Hesitations are marked based on specific conditions related to the magnitude and 
    continuity of the yaw angle changes.

    Then, the detected turns are characterized by their start and end times, duration, angle of turn, peak 
    angular velocity, and direction (left or right). Turns with angles equal to or greater than 90 degrees 
    and durations between 0.5 and 10 seconds are selected for further analysis. Finally, the detected turns 
    along with their characteristics (onset, duration, direction, etc.) are stored in a pandas DataFrame 
    (detected_turns attribute).

    Optionally, if plot_results is set to True, the algorithm generates a plot visualizing the accelerometer 
    and gyroscope data alongside the detected turns. This visualization aids in the qualitative assessment of 
    the algorithm's performance and provides insights into the dynamics of the detected turns.

    Attributes:
        gyro_convert_unit (float): Conevrsion of gyro data unit from deg/s to rad/s.
        tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
        tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.

    Methods:
        detect(data, sampling_freq_Hz):
            Detects turns using accelerometer and gyro signals.

            Args:
                data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
                sampling_freq_Hz (float, int): Sampling frequency of the signals in Hz.
                plot_results (bool, optional): If True, generates a plot. Default is False.

            Returns:
                PhamTurnDetection: an instance of the class with the detected turns
                stored in the 'detected_turns' attribute.

    Examples:
        >>> pham = PhamTurnDetection()
        >>> pham.detect(
                data=input_data,
                sampling_freq_Hz=200.0,
                )
        >>> print(pham.detected_turns)
                onset   duration   event_type   direction_of_turn   angle of turn (deg)   peak angular velocity (deg/s)   tracking_systems    tracked_points
            0   4.04    3.26       turn         left               -197.55                159.45                          imu                 LowerBack
            1   9.44    3.35       turn         right               199.69                144.67                          imu                 LowerBack

    References:
        [1] Pham et al. (2017). Algorithm for Turning Detection and Analysis Validated under Home-Like Conditions...
    """

    def __init__(
        self,
        gyro_convert_unit: float = np.pi/180,
        tracking_systems: str = "imu",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the PhamTurnDetection instance.

        Args:
            gyro_convert_unit (float): Conevrsion of gyro data unit from deg/s to rad/s.
            tracking_systems (str, optional): Tracking systems used. Default is 'imu'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.gyro_convert_unit = gyro_convert_unit
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
            which is a pandas DataFrame in BIDS format with the following information:
                - onset: Start time of the turn [s].
                - duration: Duration of the turn [s].
                - event_type: Type of the event (turn).
                - direction_of_turn: Direction of turn which is either left or right.
                - angle_of_turn_deg: Angle of the turn in degrees.
                - peak_angular_velocity: Peak angular velocity [deg/s].
                - tracking_systems: Tracking systems used (default is 'imu').
                - tracked_points: Tracked points on the body (default is 'LowerBack').
        """
        # Select acceleration data and convert it to numpy array format
        accel = data.iloc[:, 0:3].copy()
        accel = accel.to_numpy()

        # Select gyro data and convert it to numpy array format
        gyro = data.iloc[:, 3:6].copy()
        gyro = gyro.to_numpy()

        # Convert gyro data unit from deg/s to rad/s
        gyro *= self.gyro_convert_unit

        # Compute the variance of the moving window of gyro signal
        gyro_vars = []

        for i in range(3):
            gyro_var = preprocessing.moving_var(data=gyro[:, i], window=sampling_freq_Hz)
            gyro_vars.append(gyro_var)

        gyro_var_1, gyro_var_2, gyro_var_3 = gyro_vars

        # Threshold value for identifying periods where the variance is low
        thr = 2 * 10**-4
        
        # Identify periods where the variance is below the threshold
        locs = (gyro_var_1 <= thr) & (gyro_var_2 <= thr) & (gyro_var_3 <= thr)
        
        # Exclude the last 'sampling_freq_Hz' samples from the identified periods
        locs[-sampling_freq_Hz:] = False
        
        # If no bias periods are found, return the original gyro values
        if np.sum(locs) == 0:
            gyro_out = gyro
            return gyro_out
        
        # Find the start and end indices of the first bias period
        locs_place = np.where(locs)[0]
        location = np.array([[locs_place[0]], [locs_place[0] + sampling_freq_Hz - 1]])
        
        # Calculate the bias for each axis within the identified period
        gyro_bias = np.mean(gyro[location[0, 0]:location[1, 0] + 1], axis=0)
        
        # Subtract the bias from the original gyro signal
        gyro = gyro - gyro_bias

        # Integrate x component of the gyro signal to get yaw angle (also convert gyro unit to deg/s)
        yaw = scipy.integrate.cumtrapz(gyro[:, 0] / self.gyro_convert_unit, initial=0) / sampling_freq_Hz

        # Find zero-crossings indices
        index_zero_crossings = np.where(np.diff(np.sign(gyro[:, 0])))[0]  

        # Calculate turns from yaw angle
        turns_all = yaw[index_zero_crossings[1:]] - yaw[index_zero_crossings[:-1]]  
        
        # Marks hesitations in the signal
        # Initialize an array to mark hesitations
        hesitation_markers = np.zeros(len(turns_all))

        # Loop through each index in the turns_all array
        for i in range(len(turns_all)):
            # Check if the absolute value of the turn angle at index i is greater than or equal to 10
            if abs(turns_all[i]) >= 10:
                # Loop to search for potential hesitations
                for j in range(i + 1, len(turns_all)):
                    # Check if the difference between current index and i exceeds 4, or if the time between zero crossings exceeds half a second
                    if (j - i) > 4 or (index_zero_crossings[j] - index_zero_crossings[i + 1] > (sampling_freq_Hz / 2)):
                        # Break the loop if the conditions for hesitation are not met
                        break
                    else:
                        # Check conditions for hesitation:
                        # - Absolute values of both turns are greater than or equal to 10
                        # - The relative change in yaw angle is less than 20% of the minimum turn angle
                        # - The signs of both turns are the same
                        if (abs(turns_all[i]) >= 10 and abs(turns_all[j]) >= 10 and
                            abs(yaw[index_zero_crossings[i + 1]] - yaw[index_zero_crossings[j]]) /
                            min(abs(turns_all[i]), abs(turns_all[j])) < 0.2 and
                            np.sign(turns_all[i]) == np.sign(turns_all[j])):
                            
                            # Mark the range between i and j (inclusive) as a hesitation
                            hesitation_markers[i:j+1] = 1
                            # Break the inner loop as the hesitation condition is met
                            break

        # Initialize variables to store data related to turns without hesitation
        sum_temp = 0  # Temporary sum for accumulating turn angles
        turns_no_hesitation = []  # List to store turn angles without hesitation
        flags_start_no_hesitation = []  # List to store start indices of turns without hesitation
        flags_end_no_hesitation = []  # List to store end indices of turns without hesitation
        durations_no_hesitation = []  # List to store durations of turns without hesitation
        z = 1  # Index for keeping track of the current turn

        # Iterate through each index in the hesitation_markers array
        for i in range(len(hesitation_markers)):
            # Check if there is a hesitation marker at the current index
            if hesitation_markers[i] == 1:
                # Check if sum_temp is zero, indicating the start of a new turn
                if sum_temp == 0:
                    f1 = index_zero_crossings[i]  # Store the start index of the turn
                
                # Check if the absolute value of the turn angle is greater than or equal to 10
                if abs(turns_all[i]) >= 10:
                    try:
                        # Check if the next index also has a hesitation marker
                        if hesitation_markers[i + 1] != 0:
                            # Iterate through subsequent indices to find the end of the turn
                            for j in range(i + 1, len(hesitation_markers)):
                                # Check if the absolute value of the turn angle is greater than or equal to 10
                                if abs(turns_all[j]) >= 10:
                                    # Check if the signs of the turn angles are the same
                                    if np.sign(turns_all[j]) == np.sign(turns_all[i]):
                                        sum_temp += turns_all[i]  # Accumulate the turn angle
                                    else:
                                        f2 = hesitation_markers[i + 1]  # Store the end index of the turn
                                        sum_temp += turns_all[i]  # Accumulate the turn angle
                                        turns_no_hesitation.append(sum_temp)  # Store the turn angle without hesitation
                                        flags_start_no_hesitation.append(f1)  # Store the start index of the turn
                                        flags_end_no_hesitation.append(f2)  # Store the end index of the turn
                                        durations_no_hesitation.append((f2 - f1) / sampling_freq_Hz)  # Calculate and store the duration of the turn
                                        z += 1  # Increment the turn index
                                        sum_temp = 0  # Reset the temporary sum
                                        del f1, f2  # Delete stored indices to avoid conflicts
                                    break  # Exit the loop once the turn is processed
                        else:
                            f2 = index_zero_crossings[i + 1]  # Store the end index of the turn
                            sum_temp += turns_all[i]  # Accumulate the turn angle
                            turns_no_hesitation.append(sum_temp)  # Store the turn angle without hesitation
                            flags_start_no_hesitation.append(f1)  # Store the start index of the turn
                            flags_end_no_hesitation.append(f2)  # Store the end index of the turn
                            durations_no_hesitation.append((f2 - f1) / sampling_freq_Hz)  # Calculate and store the duration of the turn
                            z += 1  # Increment the turn index
                            del f1, f2  # Delete stored indices to avoid conflicts
                            sum_temp = 0  # Reset the temporary sum
                    except:
                        f2 = index_zero_crossings[i + 1]  # Store the end index of the turn
                        sum_temp += turns_all[i]  # Accumulate the turn angle
                        turns_no_hesitation.append(sum_temp)  # Store the turn angle without hesitation
                        flags_start_no_hesitation.append(f1)  # Store the start index of the turn
                        flags_end_no_hesitation.append(f2)  # Store the end index of the turn
                        durations_no_hesitation.append((f2 - f1) / sampling_freq_Hz)  # Calculate and store the duration of the turn
                        z += 1  # Increment the turn index
                        del f1, f2  # Delete stored indices to avoid conflicts
                        sum_temp = 0  # Reset the temporary sum
                else:
                    sum_temp += turns_all[i]  # Accumulate the turn angle if it's smaller than 10 degrees
            else:  # If there's no hesitation marker at the current index
                turns_no_hesitation.append(turns_all[i])  # Store the turn angle without hesitation
                flags_start_no_hesitation.append(index_zero_crossings[i])  # Store the start index of the turn
                flags_end_no_hesitation.append(index_zero_crossings[i + 1])  # Store the end index of the turn
                durations_no_hesitation.append((index_zero_crossings[i + 1] - index_zero_crossings[i]) / sampling_freq_Hz)  # Calculate and store the duration of the turn
                z += 1  # Increment the turn index

        # Initialize lists to store information about turns >= 90 degrees
        turns_90 = []  # List to store turn angles >= 90 degrees
        flags_start_90 = []  # List to store start indices of turns >= 90 degrees
        flags_end_90 = []  # List to store end indices of turns >= 90 degrees

        # Iterate through each turn without hesitation
        for k in range(len(turns_no_hesitation)):
            # Check if the turn angle is greater than or equal to 90 degrees
            # and if the duration of the turn is between 0.5 and 10 seconds
            if (abs(turns_no_hesitation[k]) >= 90 and durations_no_hesitation[k] >= 0.5 and durations_no_hesitation [k] < 10):
                # If conditions are met, store information about the turn
                turns_90.append(turns_no_hesitation[k])  # Store the turn angle
                flags_start_90.append(flags_start_no_hesitation[k])  # Store the start index of the turn
                flags_end_90.append(flags_end_no_hesitation[k])  # Store the end index of the turn

        # Initialize lists to store additional information about >= 90 degree turns
        duration_90 = []  # List to store durations of >= 90 degree turns
        peak_angular_velocities = []  # List to store peak angular velocities of >= 90 degree turns
        angular_velocity_start = []  # List to store angular velocities at the start of >= 90 degree turns
        angular_velocity_end = []  # List to store angular velocities at the end of >= 90 degree turns
        angular_velocity_middle = []  # List to store angular velocities in the middle of >= 90 degree turns
        diff_yaw = np.diff(yaw)  # Compute difference in yaw angle

        # Calculate additional information for each >= 90 degree turn
        for k in range(len(flags_start_90)):
            # Compute duration of the turn in seconds
            duration_nsamples = flags_end_90[k] - flags_start_90[k]
            duration_90.append(duration_nsamples / sampling_freq_Hz)
            
            # Calculate peak angular velocity during the turn
            diff_vector = abs(diff_yaw[(flags_start_90[k] - 1):(flags_end_90[k] - 1)])
            peak_angular_velocities.append(np.max(diff_vector) * sampling_freq_Hz)
            
            # Calculate average angular velocity at the start of the turn
            turn_10_percent = round(duration_nsamples * 0.1)
            angular_velocity_start.append(np.mean(abs(diff_yaw[flags_start_90[k]:(flags_start_90[k] + turn_10_percent)])) * sampling_freq_Hz)
            
            # Calculate average angular velocity at the end of the turn
            md = flags_start_90[k] + np.floor((flags_end_90[k] - flags_start_90[k]) / 2)
            angular_velocity_end.append(np.mean(abs(diff_yaw[(flags_end_90[k] - turn_10_percent):flags_end_90[k] - 1])) * sampling_freq_Hz)
            
            # Calculate average angular velocity in the middle of the turn
            turn_5_percent = round(duration_nsamples * 0.05)
            md = int(md)  # Convert md to an integer
            angular_velocity_middle.append(np.mean(abs(diff_yaw[int(md - turn_5_percent):int(md + turn_5_percent)])) * sampling_freq_Hz)

        # Determine direction of the turn (left or right)
        direction_of_turns = []  # Initialize list to store directions

        for turn_angle in turns_90:
            if turn_angle < 0:
                direction_of_turns.append('left')
            else:
                direction_of_turns.append('right')

        # Create a DataFrame with postural transition information
        detected_turns = pd.DataFrame(
            {
                "onset": np.array(flags_start_90) / sampling_freq_Hz,
                "duration": duration_90,
                "event_type": "turn",
                'direction_of_turn': direction_of_turns,
                "angle_of_turn_deg": turns_90,
                "peak_angular_velocity": peak_angular_velocities,
                "tracking_systems": self.tracking_systems,
                "tracked_points": self.tracked_points,
            }
        )

        # Assign the DataFrame to the 'detected_turns' attribute
        self.detected_turns = detected_turns

        # If Plot_results set to true
        if plot_results:

            preprocessing.pham_turn_plot_results(
                accel, gyro/(self.gyro_convert_unit), detected_turns, sampling_freq_Hz
            )

        # Return an instance of the class
        return self