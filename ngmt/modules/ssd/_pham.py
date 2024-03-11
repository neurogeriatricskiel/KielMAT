# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.utils import quaternion
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
                plot_results (bool, optional): If True, generates a plot. Default is False.
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

        Examples:
            Determines sit to stand and stand to sit in the sensor signal.

            >>> pham = PhamSittoStandStandtoSitDetection()
            >>> pham.detect(
                    data=,
                    sampling_freq_Hz=200,
                    )
            >>> postural_transitions = pham.postural_transitions_
            >>> print(postural_transitions)
                    onset      duration    event_type       postural transition angle [°]   maximum flexion velocity [°/s]  maximum extension velocity [°/s]  tracking_systems    tracked_points
                0   17.895     1.800       sit to stand     53.263562                       79                              8                                 imu                 LowerBack  
                1   54.655     1.905       stand to sit     47.120448                       91                              120                               imu                 LowerBack  
                2   56.020     1.090       sit to stand     23.524748                       62                              10                                imu                 LowerBack  
                3   135.895    2.505       stand to sit     21.764146                       40                              65                                imu                 LowerBack  

        References:
            [1] Pham et al. (2018). Validation of a Lower Back "Wearable"-Based Sit-to-Stand and 
            Stand-to-Sit Algorithm for Patients With Parkinson's Disease and Older Adults in a Home-Like 
            Environment. Frontiers in Neurology, 9, 652. https://doi.org/10.3389/fneur.2018.00652

    """

    def __init__(
        self,
        cutoff_freq_hz: float = 5.0,
        accel_convert_unit = 9.81,
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
        self, data: pd.DataFrame, sampling_freq_Hz: float, plot_results: bool = False
    ) -> pd.DataFrame:
        """
        Detects sit to stand and stand to sit based on the input acceleromete and gyro data.

        Args:
            data (pd.DataFrame): Input accelerometer and gyro data (N, 6) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the input data.
            plot_results (bool, optional): If True, generates a plot. Default is False.

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
        tilt_angle_rad = np.deg2rad(tilt_angle_deg)

        # Calculate sine of the tilt angle in radians
        tilt_sin = np.sin(tilt_angle_rad)

        # Apply wavelet decomposition with level of 3
        tilt_dwt_3 = preprocessing.wavelet_decomposition(data = tilt_sin, level = 3, wavetype='coif5')

        # Apply wavelet decomposition with level of 10
        tilt_dwt_10 = preprocessing.wavelet_decomposition(data = tilt_sin, level = 10, wavetype='coif5')

        # Calculate difference 
        tilt_dwt = tilt_dwt_3 - tilt_dwt_10

        # Find peaks in denoised tilt signal:  peaks of the tilt_dwt signal with magnitude and prominence >0.1 were defined as PT events
        local_peaks, _ = scipy.signal.find_peaks(tilt_dwt, height=0.2, prominence=0.2)

        # Calculate the norm of acceleration
        acceleration_norm = np.sqrt(accel[:,0] ** 2 + accel[:,1] ** 2 + accel[:,2] ** 2)

        # Calculate absolute value of the acceleration signal
        acceleration_norm = np.abs(acceleration_norm)
       
        # Detect stationary parts of the signal based on the deifned threshold
        stationary_1 = acceleration_norm < 0.05
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

        # Set initial period and check if enough stationary data is available
        # Stationary periods are defined as the episodes when the lower back of the participant was almost not moving and not rotating.
        # The thresholds are determined based on the training data set.
        init_period = 0.1

        if np.sum(stationary[:int(sampling_freq_Hz*init_period)]) >= 200: # If the process is stationary in the first 2s
                # If there is enough stationary data, perform sensor fusion using accelerometer and gyro data
                # Initialize quaternion array for orientation estimation
                quat = np.zeros((len(time), 4))

                # Initial convergence: Update the quaternion using the mean accelerometer values over a certain period
                # This helps in initializing the orientation for accurate estimation
                index_sel = np.arange(0, np.where(time >= time[0] + init_period)[0][0] + 1)
                mean_accel = np.mean(accel[index_sel], axis=0)
                quat[0] = quaternion.rotm2quat(np.eye(3) + quaternion.axang2rotm(mean_accel))

                # Update the quaternion for all data points
                for t in range(1, len(time)):
                    # Calculate the rotation matrix from gyroscope data
                    dt = time[t] - time[t-1]
                    ang_velocity = gyro[t] * dt
                    delta_rot = quaternion.axang2rotm(ang_velocity)

                    # Update the quaternion based on the rotation matrix
                    quat[t] = quaternion.quatmultiply(quat[t - 1], quaternion.rotm2quat(delta_rot))

                    # Normalize the quaternion to avoid drift
                    quat[t] = quaternion.quatnormalize(quat[t])
            
                # Analyze gyro data to detect peak velocities and directional changes
                # Zero-crossing method is used to define the beginning and the end of a PT in the gyroscope signal
                iZeroCr = np.where((gyro[:,1][:-1] * gyro[:,1][1:]) < 0)[0]

                # Calculate the difference between consecutive values
                gyrY_diff = np.diff(gyro[:,1])

                # Beginning of a PT was defined as the first zero crossing point of themedio-lateral angular 
                # velocity (gyro[:,1]) on the left side of the PT event, with negative slope.
                # Initialize left side indices with ones
                ls = np.ones_like(local_peaks)

                # Initialize right side indices with length of gyro data
                # rs = len(gyro[:,1]) * np.ones_like(local_peaks)
                rs = np.full_like(local_peaks, len(gyro[:,1]))
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
                            # Store the index of the left side
                            ls[i] = pt + dist2peak_ls[j]
                            break

                # Further analysis to distinguish between different types of postural transitions (sit-to-stand or stand-to-sit)
                # Rotate body accelerations to Earth frame
                acc = quaternion.rotm2quat(np.column_stack((accel[:,0], accel[:,1], accel[:,2])), quat)
                
                # Remove gravity from measurements
                acc -= np.array([[0, 0, 1]] * len(time))

                # Convert acceletion data to m/s^2
                acc *= self.accel_convert_unit
                
                # Calculate velocities
                vel = np.zeros_like(acc)

                # Iterate over time steps
                for t in range(1, len(vel)):
                    # Integrate acceleration to calculate velocity
                    vel[t, :] = vel[t - 1, :] + acc[t, :] * sampling_period
                    if stationary[t] == 1:
                        # Force zero velocity when stationary
                        vel[t, :] = [0, 0, 0]

                # Compute and remove integral drift
                velDrift = np.zeros_like(vel)

                # Indices where stationary changes to non-stationary
                activeStart = np.where(np.diff(stationary) == -1)[0]

                # Indices where non-stationary changes to stationary
                activeEnd = np.where(np.diff(stationary) == 1)[0]
                if activeStart[0] > activeEnd[0]:
                    # Ensure start from index 0 if starts non-stationary
                    activeStart = np.insert(activeStart, 0, 0)

                if activeStart[-1] > activeEnd[-1]:
                    # Ensure last segment ends properly
                    activeEnd = np.append(activeEnd, len(stationary))
                for i in range(len(activeEnd)):
                    # Calculate drift rate
                    driftRate = vel[activeEnd[i] - 1] / (activeEnd[i] - activeStart[i])

                    # Enumerate time steps within the segment
                    enum = np.arange(1, activeEnd[i] - activeStart[i] + 1)
                    
                    # Calculate drift for each time step
                    drift = np.column_stack((enum * driftRate[0], enum * driftRate[1], enum * driftRate[2]))

                    # Store the drift for this segment
                    velDrift[activeStart[i]:activeEnd[i], :] = drift
                
                # Remove integral drift from velocity
                vel -= velDrift
                
                # Compute translational position
                pos = np.zeros_like(vel)

                 # Iterate over time steps
                for t in range(1, len(pos)):
                    # Integrate velocity to yield position
                    pos[t, :] = pos[t - 1, :] + vel[t, :] * sampling_period 

                # Estimate vertical displacement and classify as actual PTs or Attempts
                # Calculate vertical displacement
                disp_z = pos[rs, 2] - pos[ls, 2]
                
                # Initialize flag for actual PTs
                pt_actual_flag = np.zeros_like(local_peaks)

                for i in range(len(disp_z)):
                    # Displacement greater than 10cm and less than 1m
                    if 0.1 < abs(disp_z[i]) < 1:
                        # Flag as actual PT if displacement meets criteria 
                        pt_actual_flag[i] = 1

                # Initialize list for PT types
                pt_type = []

                # Distinguish between different types of postural transitions
                for i in range(len(local_peaks)):
                    if pt_actual_flag[i] == 1:
                        if disp_z[i] == 0:
                            pt_type.append('NA')
                        elif disp_z[i] > 0:
                            pt_type.append('sit to stand')
                        else:
                            pt_type.append('stand to sit')
                    else:
                        pt_type.append('NA')

                # Calculate maximum flexion velocity and maximum extension velocity
                flexion_max_vel = np.zeros_like(local_peaks)
                extension_max_vel = np.zeros_like(local_peaks)
                for i in range(len(local_peaks)):
                    flexion_max_vel[i] = max(abs(gyro[:,1][ls[i]:local_peaks[i]]))
                    extension_max_vel[i] = max(abs(gyro[:,1][local_peaks[i]:rs[i]]))

                # Calculate PT angle
                pt_angle = np.abs(tilt_angle_deg[local_peaks] - tilt_angle_deg[ls])
                if ls[0] == 0:
                    # Adjust angle for the first PT if necessary
                    pt_angle[0] = np.abs(tilt_angle_deg[local_peaks[0]] - tilt_angle_deg[rs[0]])

                # Calculate duration of each PT
                duration = (rs - ls) / sampling_freq_Hz

                # Convert peak times to integers
                time_pt = time[local_peaks]

                # Initialize PTs list
                # i.e., the participant was considered to perform a complete standing up or sitting down movement
                PTs = [['Time[s]', 'Type', 'Angle[°]', 'Duration[s]', 'Max flexion velocity[°/s]',
                        'Max extension velocity[°/s]', 'Vertical displacement[m]']]
                
                # Initialize Attempts list
                # i.e., the participant was considered not to perform a complete PT, e.g., forward and backwards body motion
                Attempts = [['Time[s]', 'Type', 'Angle[°]', 'Duration[s]', 'Max flexion velocity[°/s]',
                            'Max extension velocity[°/s]', 'Vertical displacement[m]']]
                
                # Iterate over detected peaks
                for i in range(len(local_peaks)):
                    if pt_actual_flag[i] == 1:
                        PTs.append([time_pt[i], pt_type[i], pt_angle[i], duration[i], flexion_max_vel[i],
                                    extension_max_vel[i], disp_z[i]]) # Append PT details to PTs list
                    else:
                        Attempts.append([time_pt[i], pt_type[i], pt_angle[i], duration[i], flexion_max_vel[i],
                                        extension_max_vel[i], disp_z[i]]) # Append PT details to Attempts list

                # Extract postural transition information from PTs
                time_pt = [pt[0] for pt in PTs[1:]]
                pt_type = [pt[1] for pt in PTs[1:]]
                pt_angle = [pt[2] for pt in PTs[1:]]
                duration = [pt[3] for pt in PTs[1:]]
                flexion_max_vel = [pt[4] for pt in PTs[1:]]
                extension_max_vel = [pt[5] for pt in PTs[1:]]

                # Create a DataFrame with postural transition information
                postural_transitions_ = pd.DataFrame({
                    'onset': time_pt,
                    'duration': duration,
                    'event_type': pt_type,
                    'postural transition angle': pt_angle,
                    'maximum flexion velocity': flexion_max_vel,
                    'maximum extension velocity': extension_max_vel,
                    'tracking_systems': self.tracking_systems,
                    'tracked_points': self.tracked_points
                })

        else:
            # Handle cases where there is not enough stationary data
            # Find indices where the product of consecutive changes sign, indicating a change in direction
            iZeroCr = np.where((gyro[:,1][:-1] * gyro[:,1][1:]) < 0)[0]

            # Calculate the difference between consecutive values
            gyrY_diff = np.diff(gyro[:,1])

            # Initialize arrays to store left and right indices for each local peak
            # Initialize left side indices with ones
            ls = np.ones_like(local_peaks)

            # Initialize right side indices with length of gyro data
            rs = len(gyro[:,1]) * np.ones_like(local_peaks)

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
                        if j > 0: # Make sure dist2peak_ls[j] exist
                            # If the left side peak is far enough or small enough
                            if (dist2peak_ls[j] - dist2peak_ls[j - 1]) >= 25 or (tilt_angle_deg[pt + dist2peak_ls[j - 1]] - tilt_angle_deg[pt + dist2peak_ls[j]]) > 1:
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
                gyro_temp = gyro[:,1][ls[i]:rs[i]]
                min_peak = np.min(gyro_temp)
                max_peak = np.max(gyro_temp)
                if (abs(min_peak) - max_peak) > 0.5:
                    pt_type.append('sit to stand')
                else:
                    pt_type.append('stand to sit')
            
            # Calculate maximum flexion velocity and maximum extension velocity
            flexion_max_vel = np.zeros_like(local_peaks)
            extension_max_vel = np.zeros_like(local_peaks)
            for i in range(len(local_peaks)):
                flexion_max_vel[i] = max(abs(gyro[:,1][ls[i]:local_peaks[i]]))
                extension_max_vel[i] = max(abs(gyro[:,1][local_peaks[i]:rs[i]]))
                
            # Calculate PT angle
            pt_angle = np.abs(tilt_angle_deg[local_peaks] - tilt_angle_deg[ls])
            if ls[0] == 1:
                pt_angle[0] = abs(tilt_angle_deg[local_peaks[0]] - tilt_angle_deg[rs[0]])

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
        flexion_max_vel = [flexion_max_vel[idx] for idx, val in enumerate(flexion_max_vel) if i[idx]]
        extension_max_vel = [extension_max_vel[idx] for idx, val in enumerate(extension_max_vel) if i[idx]]

        # Create a DataFrame with postural transition information
        postural_transitions_ = pd.DataFrame({
            'onset': time_pt,
            'duration': duration,
            'event_type': pt_type,
            'postural transition angle': pt_angle,
            'maximum flexion velocity': flexion_max_vel,
            'maximum extension velocity': extension_max_vel,
            'tracking_systems': self.tracking_systems,
            'tracked_points': self.tracked_points
        })

        # Assign the DataFrame to the 'postural_transitions_' attribute
        self.postural_transitions_ = postural_transitions_

        # If Plot_results set to true
        if plot_results:

            preprocessing.pham_plot_results(accel, gyro, postural_transitions_, sampling_freq_Hz)

        # Return an instance of the class
        return self