# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


def Gait_Sequence_Detection(imu_acceleration, sampling_frequency, plot_results):
    """
    Perform Gait Sequence Detection (GSD) using low back accelerometer data.

    Args:
        imu_acceleration (numpy.ndarray): Input accelerometer data (N, 3) for x, y, and z axes.
        sampling_frequency (float): Sampling frequency of the accelerometer data.

    Returns:
        list: A list of dictionaries containing gait sequence information, including start and end times, and sampling frequency.

    Description:
        This function performs Gait Sequence Detection (GSD) on accelerometer data
        collected from a low back sensor. The purpose of GSD is to identify and
        characterize walking bouts or gait sequences within the input data.

        The input accelerometer data should be provided as a numpy.ndarray with shape
        (N, 3), where N is the number of data points. The three columns represent the
        acceleration along the x, y, and z axes.

        The function processes the input data through a series of signal processing steps
        including resampling, filtering, wavelet transform, and peak detection to identify
        gait sequences. The detected gait sequences are returned as a list of dictionaries
        containing start and end times.

        If `plot_results` is set to True, the function will also generate a plot showing
        the pre-processed acceleration data and the detected gait sequences for
        visualization purposes.
    """
    # Error handling for invalid input data
    if not isinstance(imu_acceleration, np.ndarray) or imu_acceleration.shape[1] != 3:
        raise ValueError("imu_acceleration must be a 2D numpy array with 3 columns.")

    if not isinstance(sampling_frequency, (int, float)) or sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be a positive float.")

    # Initialize the GSD_Output dictionary
    GSD_Output = {}

    # Convert imu_acceleration to a numpy array of float64 data type
    imu_acceleration = np.array(imu_acceleration, dtype=np.float64)

    # Calculate the norm of acceleration as acceleration_norm using x, y, and z components
    acceleration_norm = np.sqrt(
        imu_acceleration[:, 0] ** 2
        + imu_acceleration[:, 1] ** 2
        + imu_acceleration[:, 2] ** 2
    )

    # Resample acceleration_norm to target sampling frequency
    initial_sampling_frequency = sampling_frequency
    target_sampling_frequency = 40
    resampled_acceleration = preprocessing.resample_interpolate(
        acceleration_norm, initial_sampling_frequency, target_sampling_frequency
    )

    # Applying Savitzky-Golay filter to smoothen the resampled data
    smoothed_acceleration = preprocessing.lowpass_filter(
        resampled_acceleration, method="savgol", window_length=21, polynomial_order=7
    )

    # Remove 40Hz drift from the filtered data
    drift_removed_acceleration = preprocessing.highpass_filter(smoothed_acceleration)

    # Filter data using the low-pass filter
    filtered_acceleration = preprocessing.fir_lowpass_filter(drift_removed_acceleration)

    # Perform the continuous wavelet transform on the filtered acceleration data
    wavelet_transform_result = preprocessing.apply_continuous_wavelet_transform(
        filtered_acceleration,
        scales=10,
        desired_scale=10,
        wavelet="gaus2",
        sampling_frequency=target_sampling_frequency,
    )

    # Applying Savitzky-Golay filter to further smoothen the wavelet transformed data
    smoothed_wavelet_result = preprocessing.lowpass_filter(
        wavelet_transform_result, window_length=11, polynomial_order=5
    )

    # Perform continuous wavelet transform
    further_smoothed_wavelet_result = preprocessing.apply_continuous_wavelet_transform(
        smoothed_wavelet_result,
        scales=10,
        desired_scale=10,
        wavelet="gaus2",
        sampling_frequency=target_sampling_frequency,
    )
    further_smoothed_wavelet_result = further_smoothed_wavelet_result.T

    # Smoothing the data using successive Gaussian filters
    filtered_signal = preprocessing.apply_successive_gaussian_filters(
        further_smoothed_wavelet_result
    )

    # Use pre-processsed signal for post-processing purposes
    detected_activity_signal = filtered_signal

    # Compute the envelope of the processed acceleration data
    envelope = []
    envelope, _ = preprocessing.calculate_envelope_activity(
        detected_activity_signal,
        int(round(target_sampling_frequency)),
        1,
        int(round(target_sampling_frequency)),
        0,
    )

    # Initialize a list for walking bouts
    walking_bouts = [0]

    # Process alarm data to identify walking bouts
    if envelope.size > 0:
        index_ranges = preprocessing.find_consecutive_groups(envelope > 0)
        for j in range(len(index_ranges)):
            if index_ranges[j, 1] - index_ranges[j, 0] <= 3 * target_sampling_frequency:
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
    min_peaks, max_peaks = preprocessing.find_local_min_max(selected_signal, threshold)

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
            filtered_walking_bouts[j]["start"] : filtered_walking_bouts[j]["end"] + 1
        ] = 1

    # Call the find_consecutive_groups function with the walking_labels variable
    ind_noWk = []
    ind_noWk = preprocessing.find_consecutive_groups(walking_labels == 0)

    # Merge walking bouts if break less than 3 seconds
    if ind_noWk.size > 0:
        for j in range(len(ind_noWk)):
            if ind_noWk[j, 1] - ind_noWk[j, 0] <= target_sampling_frequency * 3:
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
                    "Start": walk[j]["start"] / target_sampling_frequency,
                    "End": walk[j]["end"] / target_sampling_frequency,
                    "fs": sampling_frequency,
                }
            )
        print("Gait sequence(s) detected.")
    else:
        print("No gait sequence(s) detected.")

    # Plot results if set to true
    if plot_results:
        plt.figure(figsize=(10, 6))
        plt.plot(
            np.arange(len(detected_activity_signal)) / (60 * target_sampling_frequency),
            detected_activity_signal,
            "b",
            linewidth=3,
        )

        plt.plot(
            np.arange(len(detected_activity_signal)) / (60 * target_sampling_frequency),
            walking_labels,
            color=cfg_colors["raw"][0],
            linewidth=5,
        )
        plt.title("Detected activity and walking labels", fontsize=20)
        plt.xlabel("Time (minutes)", fontsize=20)

        plt.ylabel("Acceleration (g)", fontsize=20)

        # Fill the area between start and end times
        for sequence in GSD_Output:
            start_time = sequence["Start"] / 60  # Convert to minutes
            end_time = sequence["End"] / 60  # Convert to minutes
            plt.axvline(start_time, color="g")
            plt.axvline(end_time, color="r")
            plt.axvspan(start_time, end_time, facecolor="grey", alpha=0.8)

        plt.legend(
            [
                "Pre-processed acceleration signal",
            ],
            fontsize=16,
        )

        plt.grid(True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    return GSD_Output
