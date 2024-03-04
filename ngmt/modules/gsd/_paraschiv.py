from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


class ParaschivIonescuGaitSequenceDetection:
    """
    Detects gait sequences in accelerometer data from a lower back sensor.

    This class implements the Paraschiv-Ionescu method for gait sequence detection, aiming to identify and
    characterize walking bouts within accelerometer data. It processes input data through signal processing steps
    including resampling, filtering, wavelet transform, and peak detection. The detected gait sequences are stored
    in the 'gait_sequences_' attribute.

    Attributes:
        target_sampling_freq_Hz (float): Target sampling frequency for resampling the data. Defaults to 40 Hz.
        event_type (str): Type of the detected event. Defaults to 'gait sequence'.
        tracking_systems (str): Tracking systems used. Defaults to 'SU'.
        tracked_points (str): Tracked points on the body. Defaults to 'LowerBack'.
        gait_sequences_ (pd.DataFrame): DataFrame containing gait sequence information in BIDS format.

    Methods:
        detect(data, sampling_freq_Hz, plot_results=False):
            Detects gait sequences in the provided accelerometer data.

            Args:
                data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
                sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
                plot_results (bool, optional): If True, generates a plot of the pre-processed data and the detected gait sequences. Defaults to False.
                dt_data (str, optional): Name of the original datetime in the input data. If original datetime is provided, the output onset will be based on that.

            Returns:
                ParaschivIonescuGaitSequenceDetection: An instance of the class with the detected gait sequences stored in the 'gait_sequences_' attribute.

    Examples:
        >>> gsd = ParaschivIonescuGaitSequenceDetection()
        >>> gsd.detect(data=acceleration_data, sampling_freq_Hz=100, plot_results=True)
        >>> print(gsd.gait_sequences_)
                onset   duration    event_type      tracking_systems    tracked_points
            0   4.500   5.25        gait sequence   SU                  LowerBack
            1   90.225  10.30       gait sequence   SU                  LowerBack

    References:
        [1] Paraschiv-Ionescu et al. (2019). Locomotion and cadence detection using a single trunk-fixed accelerometer...
        [2] Paraschiv-Ionescu et al. (2020). Real-world speed estimation using single trunk IMU...
    """

    def __init__(
        self,
        target_sampling_freq_Hz: float = 40.0,
        event_type: str = "gait sequence",
        tracking_systems: str = "SU",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the ParaschivIonescuGaitSequenceDetection instance.

        Args:
            target_sampling_freq_Hz (float, optional): Target sampling frequency for resampling the data. Default is 40.
            event_type (str, optional): Type of the detected event. Default is 'gait sequence'.
            tracking_systems (str, optional): Tracking systems used. Default is 'SU'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.target_sampling_freq_Hz = target_sampling_freq_Hz
        self.event_type = event_type
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.gait_sequences_ = None

    def detect(
        self,
        data: pd.DataFrame,
        sampling_freq_Hz: float,
        plot_results: bool = False,
        dt_data: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Detects gait sequences based on the input accelerometer data.

        Args:
            data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
            plot_results (bool, optional): If True, generates a plot showing the pre-processed acceleration data
            and the detected gait sequences. Default is False.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.

        Returns:
            ParaschivIonescuGaitSequenceDetection: Returns an instance of the class.
                The gait sequence information is stored in the 'gait_sequences_' attribute,
                which is a pandas DataFrame in BIDS format with the following columns:
                    - onset: Start time of the gait sequence.
                    - duration: Duration of the gait sequence.
                    - event_type: Type of the event (default is 'gait sequence').
                    - tracking_systems: Tracking systems used (default is 'SU').
                    - tracked_points: Tracked points on the body (default is 'LowerBack').
        """
        # Error handling for invalid input data
        if not isinstance(data, pd.DataFrame) or data.shape[1] != 3:
            raise ValueError(
                "Input accelerometer data must be a DataFrame with 3 columns for x, y, and z axes."
            )

        if not isinstance(sampling_freq_Hz, (int, float)) or sampling_freq_Hz <= 0:
            raise ValueError("Sampling frequency must be a positive float.")

        if not isinstance(plot_results, bool):
            raise ValueError("Plot results must be a boolean (True or False).")

        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (not isinstance(
            dt_data, pd.Series
        ) or not pd.api.types.is_datetime64_any_dtype(dt_data)):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Calculate the norm of acceleration
        acceleration_norm = np.linalg.norm(data, axis=1)

        # Resample acceleration_norm to target sampling frequency
        initial_sampling_frequency = sampling_freq_Hz
        resampled_acceleration = preprocessing.resample_interpolate(
            acceleration_norm, initial_sampling_frequency, self.target_sampling_freq_Hz
        )

        # Applying low-pass Savitzky-Golay filter to smoothen the resampled data
        smoothed_acceleration = preprocessing.lowpass_filter(
            resampled_acceleration,
            method="savgol",
            window_length=21,
            polynomial_order=7,
        )

        # Remove 40Hz drift from the filtered data
        drift_removed_acceleration = preprocessing.highpass_filter(
            signal=smoothed_acceleration,
            sampling_frequency=self.target_sampling_freq_Hz,
            method="iir",
        )

        # Filter data using the fir low-pass filter
        filtered_acceleration = preprocessing.lowpass_filter(
            drift_removed_acceleration, method="fir"
        )

        # Perform the continuous wavelet transform on the filtered acceleration data
        wavelet_transform_result = preprocessing.apply_continuous_wavelet_transform(
            filtered_acceleration,
            scales=10,
            desired_scale=10,
            wavelet="gaus2",
            sampling_frequency=self.target_sampling_freq_Hz,
        )

        # Applying Savitzky-Golay filter to further smoothen the wavelet transformed data
        smoothed_wavelet_result = preprocessing.lowpass_filter(
            wavelet_transform_result, window_length=11, polynomial_order=5
        )

        # Perform continuous wavelet transform
        further_smoothed_wavelet_result = (
            preprocessing.apply_continuous_wavelet_transform(
                smoothed_wavelet_result,
                scales=10,
                desired_scale=10,
                wavelet="gaus2",
                sampling_frequency=self.target_sampling_freq_Hz,
            )
        )
        further_smoothed_wavelet_result = further_smoothed_wavelet_result.T

        # Smoothing the data using successive Gaussian filters
        filtered_signal = preprocessing.apply_successive_gaussian_filters(
            further_smoothed_wavelet_result
        )

        # Use pre-processsed signal for post-processing purposes
        detected_activity_signal = filtered_signal

        # Compute the envelope of the processed acceleration data
        envelope, _ = preprocessing.calculate_envelope_activity(
            detected_activity_signal,
            int(round(self.target_sampling_freq_Hz)),
            1,
            int(round(self.target_sampling_freq_Hz)),
            0,
        )

        # Initialize a list for walking bouts
        walking_bouts = [0]

        # Process alarm data to identify walking bouts
        if envelope.size > 0:
            index_ranges = preprocessing.find_consecutive_groups(envelope > 0)
            for j in range(len(index_ranges)):
                if (
                    index_ranges[j, 1] - index_ranges[j, 0]
                    <= 3 * self.target_sampling_freq_Hz
                ):
                    envelope[index_ranges[j, 0] : index_ranges[j, 1] + 1] = 0
                else:
                    walking_bouts.extend(
                        detected_activity_signal[
                            index_ranges[j, 0] : index_ranges[j, 1] + 1
                        ]
                    )

            # Convert walk_low_back list to a NumPy array
            walking_bouts_array = np.array(walking_bouts)

            # Find positive peaks in the walk_low_back_array
            positive_peak_indices, _ = scipy.signal.find_peaks(walking_bouts_array)

            # Get the corresponding y-axis data values for the positive peak
            positive_peaks = walking_bouts_array[positive_peak_indices]

            # Find negative peaks in the inverted walk_low_back array
            negative_peak_indices, _ = scipy.signal.find_peaks(-walking_bouts_array)

            # Get the corresponding y-axis data values for the positive peak
            negative_peaks = -walking_bouts_array[negative_peak_indices]

            # Combine positive and negative peaks
            combined_peaks = [x for x in positive_peaks if x > 0] + [
                x for x in negative_peaks if x > 0
            ]

            # Calculate the data adaptive threshold using the 5th percentile of the combined peaks
            threshold = np.percentile(combined_peaks, 5)

            # Set selected_signal to detected_activity_signal
            selected_signal = detected_activity_signal

        else:
            threshold = 0.15
            selected_signal = smoothed_wavelet_result

        # Detect mid-swing peaks
        min_peaks, max_peaks = preprocessing.find_local_min_max(
            selected_signal, threshold
        )

        # Find pulse trains in max_peaks and remove ones with steps less than 4
        pulse_trains_max = preprocessing.identify_pulse_trains(max_peaks)

        # Access the fields of the struct-like array
        pulse_trains_max = [train for train in pulse_trains_max if train["steps"] >= 4]

        # Find pulse trains in min_peaks and remove ones with steps less than 4
        pulse_trains_min = preprocessing.identify_pulse_trains(min_peaks)

        # Access the fields of the struct-like array
        pulse_trains_min = [train for train in pulse_trains_min if train["steps"] >= 4]

        # Convert t1 and t2 to sets and find their intersection
        walking_periods = preprocessing.find_interval_intersection(
            preprocessing.convert_pulse_train_to_array(pulse_trains_max),
            preprocessing.convert_pulse_train_to_array(pulse_trains_min),
        )

        # Check if walking_periods is empty
        if walking_periods is None:
            walking_bouts = []
            MidSwing = []
        else:
            # Call the organize_and_pack_results function with walking_periods and MaxPeaks
            walking_bouts, MidSwing = preprocessing.organize_and_pack_results(
                walking_periods, max_peaks
            )
            if walking_bouts:
                # Update the start value of the first element
                walking_bouts[0]["start"] = max([1, walking_bouts[0]["start"]])

                # Update the end value of the last element
                walking_bouts[-1]["end"] = min(
                    [walking_bouts[-1]["end"], len(detected_activity_signal)]
                )

        # Calculate the length of walking bouts
        walking_bouts_length = len(walking_bouts)

        # Initialize an empty list
        filtered_walking_bouts = []

        # Initialize a counter variable "counter"
        counter = 0

        for j in range(walking_bouts_length):
            if walking_bouts[j]["steps"] >= 5:
                counter += 1
                filtered_walking_bouts.append(
                    {"start": walking_bouts[j]["start"], "end": walking_bouts[j]["end"]}
                )

        # Initialize an array of zeros with the length of detected_activity_signal
        walking_labels = np.zeros(len(detected_activity_signal))

        # Calculate the length of the filtered_walking_bouts
        filtered_walking_bouts_length = len(filtered_walking_bouts)

        for j in range(filtered_walking_bouts_length):
            walking_labels[
                filtered_walking_bouts[j]["start"] : filtered_walking_bouts[j]["end"]
                + 1
            ] = 1

        # Call the find_consecutive_groups function with the walking_labels variable
        ind_noWk = []
        ind_noWk = preprocessing.find_consecutive_groups(walking_labels == 0)

        # Merge walking bouts if break less than 3 seconds
        if ind_noWk.size > 0:
            for j in range(len(ind_noWk)):
                if ind_noWk[j, 1] - ind_noWk[j, 0] <= self.target_sampling_freq_Hz * 3:
                    walking_labels[ind_noWk[j, 0] : ind_noWk[j, 1] + 1] = 1

        # Merge walking bouts if break less than 3 seconds
        ind_Wk = []
        walkLabel_1_indices = np.where(walking_labels == 1)[0]

        if walkLabel_1_indices.size > 0:
            ind_Wk = preprocessing.find_consecutive_groups(walking_labels == 1)
            # Create an empty list to store 'walk' dictionaries
            walk = []
            if ind_Wk.size > 0:
                for j in range(len(ind_Wk)):
                    walk.append({"start": (ind_Wk[j, 0]), "end": ind_Wk[j, 1]})

            n = len(walk)
            GSD_Output = []

            for j in range(n):
                GSD_Output.append(
                    {
                        "Start": walk[j]["start"] / self.target_sampling_freq_Hz,
                        "End": walk[j]["end"] / self.target_sampling_freq_Hz,
                        "fs": sampling_freq_Hz,
                    }
                )
            print(f"{n} gait sequence(s) detected.")
        else:
            print("No gait sequence(s) detected.")

        # Create a DataFrame from the gait sequence data
        gait_sequences_ = pd.DataFrame(GSD_Output)
        gait_sequences_["onset"] = gait_sequences_["Start"]
        gait_sequences_["duration"] = gait_sequences_["End"] - gait_sequences_["Start"]
        gait_sequences_["event_type"] = self.event_type
        gait_sequences_["tracking_systems"] = self.tracking_systems
        gait_sequences_["tracked_points"] = self.tracked_points

        # Check if the indices in ind_Wk are within the range of dt_data's index
        if ind_Wk.size > 0 and dt_data is not None:
            valid_indices = [index for index in ind_Wk[:, 0] if index < len(dt_data)]
            invalid_indices = len(ind_Wk[:, 0]) - len(valid_indices)

            if invalid_indices > 0:
                print(f"Warning: {invalid_indices} invalid index/indices found.")

            # Only use valid indices to access dt_data
            valid_dt_data = dt_data.iloc[valid_indices]

            # Create a DataFrame from the gait sequence data
            gait_sequences_ = pd.DataFrame(GSD_Output)
            gait_sequences_["onset"] = gait_sequences_["Start"]
            gait_sequences_["duration"] = (
                gait_sequences_["End"] - gait_sequences_["Start"]
            )
            gait_sequences_["event_type"] = self.event_type
            gait_sequences_["tracking_systems"] = self.tracking_systems
            gait_sequences_["tracked_points"] = self.tracked_points

            # If original datetime is available, update the 'onset' column
            gait_sequences_["onset"] = valid_dt_data.reset_index(drop=True)

        # Create a DataFrame from the gait sequence data
        gait_sequences_ = gait_sequences_[
            ["onset", "duration", "event_type", "tracking_systems", "tracked_points"]
        ]

        # Return gait_sequences_ as an output
        self.gait_sequences_ = gait_sequences_

        # currently no plotting for datetime values
        if dt_data is not None and plot_results:
            print("No plotting for datetime values.")
            plot_results = False
            return self

        # Plot results if set to true
        if plot_results:
            plt.figure(figsize=(22, 14))
            plt.plot(
                np.arange(len(detected_activity_signal))
                / (60 * self.target_sampling_freq_Hz),
                detected_activity_signal,
                label="Pre-processed acceleration signal",
            )
            plt.title("Detected gait sequences", fontsize=20)
            plt.xlabel("Time (minutes)", fontsize=20)
            plt.ylabel("Acceleration (g)", fontsize=20)

            # Fill the area between start and end times
            for index, sequence in gait_sequences_.iterrows():
                onset = sequence["onset"] / 60  # Convert to minutes
                end_time = (
                    sequence["onset"] + sequence["duration"]
                ) / 60  # Convert to minutes
                plt.axvline(onset, color="g")
                plt.axvspan(onset, end_time, facecolor="grey", alpha=0.8)
            plt.legend(
                ["Pre-processed acceleration signal", "Gait onset", "Gait duration"],
                fontsize=20,
                loc="best",
            )
            plt.grid(visible=None, which="both", axis="both")
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.show()

        return self
