# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage
import pywt
from ngmt.utils import preprocessing


def gsd_low_back_acc(imu_acceleration, sampling_frequency, plot_results):
    """
    Perform Gait Sequence Detection (GSD) using low back accelerometer data.

    Args:
        imu_acceleration (numpy.ndarray): Input accelerometer data (N, 3) for x, y, and z axes.
        sampling_frequency (float): Sampling frequency of the accelerometer data.

    Returns:
        list: A list of dictionaries containing gait sequence information, including start and end times, and sampling frequency.
    """
    GSD_Output = {}

    # Calculate the norm of acceleration as acceleration_norm using x, y, and z components.
    acceleration_norm = np.sqrt(
        imu_acceleration[:, 0] ** 2
        + imu_acceleration[:, 1] ** 2
        + imu_acceleration[:, 2] ** 2
    )

    # Resample acceleration_norm to target sampling frequency using resample_interpolate function.
    initial_sampling_frequency = (
        sampling_frequency  # Initial sampling frequency of the acceleration data
    )
    target_sampling_frequency = (
        40  # Targeted sampling frequency of the acceleration data
    )
    resampled_acceleration = preprocessing.resample_interpolate(
        acceleration_norm, initial_sampling_frequency, target_sampling_frequency
    )  # Resampled data with 40Hz

    # Applying Savitzky-Golay filter to smoothen the resampled data with frequency of 40Hz
    window_length = 21
    polynomial_order = 7
    smoothed_acceleration = scipy.signal.savgol_filter(
        resampled_acceleration, window_length, polynomial_order
    )

    # Load FIR filter designed and apply for the low SNR, impaired, asymmetric, and slow gait
    filtering_file = scipy.io.loadmat(
        "C:\\Users\\Project\\Desktop\\Gait_Sequence\\Mobilise-D-TVS-Recommended-Algorithms\\GSDB\\Library\\FIR-2-3Hz-40.mat"
    )
    num = filtering_file["Num"][0, :]

    # Remove drifts using defined function in utls (RemoveDrift40Hz).
    # Define parameters of the filter
    numerator_coefficient = num
    denominator_coefficient = np.array(
        [
            1.0,
        ]
    )
    detrended_acceleration = scipy.signal.filtfilt(
        numerator_coefficient,
        denominator_coefficient,
        preprocessing.remove_40Hz_drift(smoothed_acceleration),
    )

    # Perform the continuous wavelet transform on the filtered acceleration data accN_filt2
    scales = 10  #  At scale=10 the wavelet is stretched by a factor of 10, making it sensitive to lower frequencies in the signal.
    wavelet = "gaus2"  #  The Gaussian wavelets ("gausP" where P is an integer between 1 and and 8) correspond to the Pth order derivatives of the function
    sampling_period = (
        1 / target_sampling_frequency
    )  #  Sampling period which is equal to 1/algorithm_target_fs
    coefficients, _ = pywt.cwt(
        detrended_acceleration, np.arange(1, scales + 1), wavelet, sampling_period
    )
    desired_scale = 10  # Choose the desired scale you want to access (1 to scales) and extract it from the coefficients
    wavelet_transform_result = coefficients[desired_scale - 1, :]

    # Applying Savitzky-Golay filter to further smoothen the wavelet transformed data
    window_length = 11
    polynomial_order = 5
    smoothed_wavelet_result = scipy.signal.savgol_filter(
        wavelet_transform_result, window_length, polynomial_order
    )

    # Perform continuous wavelet transform
    coefficients, _ = pywt.cwt(
        smoothed_wavelet_result, np.arange(1, scales + 1), wavelet, sampling_period
    )
    desired_scale = 10  # Choose the desired scale you want to access (1 to scales) and extract it from the coefficients
    further_smoothed_wavelet_result = coefficients[desired_scale - 1, :]

    # Smoothing the data using successive Gaussian filters from scipy.ndimage
    sigma_1 = 1.9038  # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt5,'gaussian',window);
    sigma_2 = 1.9038  # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt6,'gaussian',window);
    sigma_3 = 2.8936  # The sigma_1 = 2.8936 gives the same results when window=15 in the MATLAB fuction smoothdata(accN_filt7,'gaussian',window);
    sigma_4 = 1.9038  # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt8,'gaussian',window);
    sigma_values = [
        sigma_1,
        sigma_2,
        sigma_3,
        sigma_4,
    ]  # Vectors of sigma values for successive Gaussian filters
    first_gaussian_filtered_signal = scipy.ndimage.gaussian_filter(
        further_smoothed_wavelet_result,
        sigma=sigma_values[0],
        order=0,
        output=None,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        radius=None,
    )
    second_gaussian_filtered_signal = scipy.ndimage.gaussian_filter(
        first_gaussian_filtered_signal,
        sigma=sigma_values[1],
        order=0,
        output=None,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        radius=None,
    )
    third_gaussian_filtered_signal = scipy.ndimage.gaussian_filter(
        second_gaussian_filtered_signal,
        sigma=sigma_values[2],
        order=0,
        output=None,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        radius=None,
    )
    fourth_gaussian_filtered_signal = scipy.ndimage.gaussian_filter(
        third_gaussian_filtered_signal,
        sigma=sigma_values[3],
        order=0,
        output=None,
        mode="reflect",
        cval=0.0,
        truncate=4.0,
        radius=None,
    )

    # Use processed acceleration data for further analysis.
    detected_activity_signal = fourth_gaussian_filtered_signal

    # Compute the envelope of the processed acceleration data.
    envelope = []
    envelope, _ = preprocessing.calculate_envelope_activity(
        detected_activity_signal,
        int(round(target_sampling_frequency)),
        1,
        int(round(target_sampling_frequency)),
        1,
    )

    # Initialize a list for walking bouts.
    walking_bouts = []

    # Process alarm data to identify walking bouts.
    if envelope.size > 0:
        non_zero_indices = np.where(envelope > 0)[0]  # Find nonzeros
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
        positive_peak_indices, _ = scipy.signal.find_peaks(
            walking_bouts_array, height=0
        )

        # Get the corresponding y-axis data values for the positive peak
        positive_peaks = walking_bouts_array[positive_peak_indices]

        # Find negative peaks in the inverted walk_low_back array
        negative_peak_indices, _ = scipy.signal.find_peaks(-walking_bouts_array)

        # Get the corresponding y-axis data values for the positive peak
        negative_peaks = -walking_bouts_array[negative_peak_indices]

        # Convert pksn list to a NumPy array before using it in concatenation
        negative_peaks_array = np.array(negative_peaks)

        # Combine positive and negative peaks
        combined_peaks = np.concatenate((positive_peaks, negative_peaks_array))

        # Calculate the data adaptive threshold using the 5th percentile of the combined peaks
        threshold = np.percentile(combined_peaks, 5)

        # Set f to sigDetActv
        selected_signal = detected_activity_signal

    else:
        threshold = (
            0.15  # If hilbert envelope fails to detect 'active', try version [1]
        )
        selected_signal = smoothed_wavelet_result

    # Detect mid-swing peaks.
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

    if walking_periods is None:  # Check if walking_periods is empty
        walking_bouts = []
        MidSwing = []
    else:
        walking_bouts, MidSwing = preprocessing.organize_and_pack_results(
            walking_periods, max_peaks
        )  # Call the organize_and_pack_results function with walking_periods and MaxPeaks
        if walking_bouts:  # Check if w is not empty
            walking_bouts[0]["start"] = max(
                [1, walking_bouts[0]["start"]]
            )  # Update the start value of the first element in w
            walking_bouts[-1]["end"] = min(
                [walking_bouts[-1]["end"], len(detected_activity_signal)]
            )  # Update the end value of the last element in w

    walking_bouts_length = len(walking_bouts)  # Calculate the length (size) of w
    filtered_walking_bouts = []  # Initialize an empty list w_new
    counter = 0  # Initialize a counter variable k
    for j in range(walking_bouts_length):  # Loop through the range from 0 to n-1
        if (
            walking_bouts[j]["steps"] >= 5
        ):  # Check if the 'steps' field of the j-th element in w is greater than or equal to 5
            counter += 1  # Increment the counter k
            filtered_walking_bouts.append(
                {"start": walking_bouts[j]["start"], "end": walking_bouts[j]["end"]}
            )  # Add a new element to w_new with 'start' and 'end' fields from the j-th element in w

    walking_labels = np.zeros(
        len(detected_activity_signal)
    )  # Initialize an array of zeros with the length of sigDetActv
    filtered_walking_bouts_length = len(
        filtered_walking_bouts
    )  # Calculate the length (size) of w_new
    for j in range(
        filtered_walking_bouts_length
    ):  # Loop through the range from 0 to n-1
        walking_labels[
            filtered_walking_bouts[j]["start"] : filtered_walking_bouts[j]["end"] + 1
        ] = 1  # Update elements in walking_labels to 1 between the 'start' and 'end' indices of the j-th element in w_new

    # Merge walking bouts if break less than 3 seconds
    ind_noWk = []
    ind_noWk = preprocessing.find_consecutive_groups(walking_labels == 0)
    if len(ind_noWk) > 0:
        for j in range(len(ind_noWk)):
            if ind_noWk[j, 1] - ind_noWk[j, 0] <= target_sampling_frequency * 3:
                walking_labels[ind_noWk[j, 0] : ind_noWk[j, 1] + 1] = 1

    ind_Wk = []
    if np.any(walking_labels == 1):
        ind_Wk = preprocessing.find_consecutive_groups(walking_labels == 1)
        if len(ind_Wk) > 0:
            GSD_Output = []
            for j in range(len(ind_Wk)):
                GSD_Output.append(
                    {
                        "Start": ind_Wk[j, 0] / target_sampling_frequency,
                        "End": ind_Wk[j, 1] / target_sampling_frequency,
                        "fs": initial_sampling_frequency,
                    }
                )
    else:
        print("No gait sequence(s) detected")

    # Plot results if set to true
    if plot_results:
        plt.figure(figsize=(10, 6))

        plt.plot(detected_activity_signal, linewidth=3)
        plt.plot(walking_labels, "r", linewidth=5)

        plt.title("Detected Activity and Walking Labels", fontsize=20)
        plt.xlabel("Samples (40Hz)", fontsize=20)
        plt.ylabel("Amplitude", fontsize=20)

        plt.legend(
            [
                "Processed Acceleration Signal from Lowerback IMU Sensor",
                "Detected Gait Sequences",
            ],
            fontsize=16,
        )
        plt.grid(True)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.show()

    return GSD_Output


import matplotlib.pyplot as plt
