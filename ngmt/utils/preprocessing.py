# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal
import scipy.io
import scipy.integrate
import scipy.ndimage
import pywt

def fir_lowpass_filter(data, fir_file="ngmt/utils/FIR_2_3Hz_40.mat"):
    """
    Apply a finite impulse response (FIR) low-pass filter to input data.

    This function loads FIR filter coefficients from a given FIR file and applies
    the filter to the input data using the `scipy.signal.filtfilt` function.

    Args:
    ----------
    data : array-like
        The input data to be filtered.
    fir_file : str, optional
        The filename of the FIR filter coefficients MAT file.
        Default is "FIR_2_3Hz_40.mat".

    Returns:
    -------
    filtered_signal : array
        The filtered signal after applying the FIR low-pass filter.

    Notes:
    -----
    The FIR filter coefficients are loaded from the specified MAT file (`fir_file`).
    The filter is applied using `scipy.signal.filtfilt`, which performs zero-phase
    filtering to avoid phase distortion.
    """
    # Remove drifts using desinged filter (remove_40Hz_drift)
    filtered_signal = remove_40Hz_drift(data)

    # Load FIR filter coefficients from the specified MAT file
    num = scipy.io.loadmat(fir_file)

    # Extract the numerator coefficients from the loaded data
    numerator_coefficient = num["Num"][0, :]

    # Define the denominator coefficients as [1.0] to perform FIR filtering
    denominator_coefficient = np.array(
        [
            1.0,
        ]
    )

    # Apply the FIR low-pass filter using filtfilt
    filtered_signal = scipy.signal.filtfilt(
        numerator_coefficient, denominator_coefficient, filtered_signal
    )

    # Return the filtered signal
    return filtered_signal


def resample_interpolate(input_signal, initial_sampling_rate, target_sampling_rate):
    """_summary_
    Resample and interpolate a signal to a new sampling rate.

    This function takes a signal `input_signal` sampled at an initial sampling rate `initial_sampling_rate`
    and resamples it to a new sampling rate `target_sampling_rate` using linear interpolation.

    Args:
    input_signal (array_like): The input signal.
    initial_sampling_rate (float): The initial sampling rate of the input signal.
    target_sampling_rate (float): The desired sampling rate for the output signal.

    Returns:
    resampled_signal (array_like): The resampled and interpolated signal.
    """
    # Calculate the length of the input signal.
    recording_time = len(input_signal)

    # Create an array representing the time indices of the input signal.
    x = np.arange(1, recording_time + 1)

    # Create an array representing the time indices of the resampled signal.
    xq = np.arange(1, recording_time + 1, initial_sampling_rate / target_sampling_rate)

    # Create an interpolation function using linear interpolation and apply it to the data.
    interpolator = scipy.interpolate.interp1d(
        x, input_signal, kind="linear", axis=0, fill_value="extrapolate"
    )

    # Resample and interpolate the input signal to the desired target sampling rate.
    resampled_signal = interpolator(xq)

    return resampled_signal


def remove_40Hz_drift(signal):
    """_summary_
    Remove 40Hz drift from a signal using a high-pass filter.

    This function applies a high-pass filter to remove low-frequency drift at 40Hz
    from the input signal `signal`.

    Args:
    signal (array_like): The input signal.

    Returns:
    filtered_signal (ndarray): The filtered signal with removed drift.
    """
    # The numerator coefficient vector of the filter.
    numerator_coefficient = np.array([1, -1])

    # The denominator coefficient vector of the filter.
    denominator_coefficient = np.array([1, -0.9748])

    # Filter signal using high-pass filter
    filtered_signal = scipy.signal.filtfilt(
        numerator_coefficient,
        denominator_coefficient,
        signal,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(numerator_coefficient), len(denominator_coefficient)) - 1),
    )

    return filtered_signal


def recursive_gaussian_smoothing(noisy_data, window_lengths, sigmas):
    """
    Apply recursive Gaussian smoothing to noisy data using different window lengths and sigmas.

    Args:
        noisy_data (numpy.ndarray): Input noisy data as a 1D NumPy array.
        window_lengths (list of int): List of window lengths for each smoothing step.
        sigmas (list of float): List of standard deviations corresponding to window_lengths.

    Returns:
        numpy.ndarray: Smoothed data after applying multiple Gaussian filters.
    """
    smoothed_data = noisy_data.copy()

    for window_length, sigma in zip(window_lengths, sigmas):
        # Create the Gaussian kernel
        x = np.arange(-window_length // 2 + 1, window_length // 2 + 1, 1)
        gaussian_kernel = np.exp(-(x**2) / (2 * sigma**2))

        # Normalize the kernel
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Apply the filter to the data using convolution
        smoothed_data = np.convolve(smoothed_data, gaussian_kernel, mode="same")

    return smoothed_data


def calculate_envelope_activity(
    input_signal, smooth_window, threshold_style, duration, plot_results
):
    """_summary_
    Calculate envelope-based activity detection using the Hilbert transform.

    This function analyzes an input signal `input_signal` to detect periods of activity based on the signal's envelope.
    It calculates the analytical signal using the Hilbert transform, smoothes the envelope, and applies an
    adaptive threshold to identify active regions.

    Args:
    input_signal (array_like): The input signal.
    smooth_window (int): Window length for smoothing the envelope.
    threshold_style (int): Threshold selection style: 0 for manual, 1 for automatic.
    duration (int): Minimum duration of activity to be detected.
    plot_results (int): Set to 1 for plotting results, 0 otherwise.

    Returns:
    tuple(ndarray, ndarray): A tuple containing:
        - alarm (ndarray): Vector indicating active parts of the signal.
        - env (ndarray): Smoothed envelope of the signal.
    """

    # Calculate the analytical signal and get the envelope
    input_signal = (
        input_signal.flatten()
    )
    # Compute the analytic signal, using the Hilbert transform form scipy.signal.
    analytic = scipy.signal.hilbert(input_signal)
    env = np.abs(analytic)  # Compute the envelope of the analytic signal.

    # Take the moving average of the analytic signal
    env = scipy.signal.convolve(
        env, np.ones(smooth_window) / smooth_window, mode="full"
    )  # Returns the discrete, linear convolution of two one-dimensional sequences.

    env = env - np.mean(env)  # Remove the offset by subtracting the mean of 'env'
    env = env / np.max(env)  # Normalize the 'env' by dividing by its maximum value

    # Threshold the signal
    if threshold_style == 0:
        plt.plot(env)
        plt.title("Select a threshold on the graph")
        THR_SIG = plt.ginput(1)[0][1]
        plt.close()
    else:
        THR_SIG = 4 * np.mean(env)

    # Set noise and signal levels
    noise = np.mean(env) / 3  # noise level
    
    # Signal level: It's used as a reference to distinguish between the background noise and the actual signal activity.
    threshold = np.mean(env)

    # Initialize Buffers
    thres_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to a threshold.
    noise_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to the noise.
    THR_buf = np.zeros(len(env))  # This buffer stores threshold values.
    alarm = np.zeros(len(env))  # This buffer tracks alarm-related information.
    h = 1

    for i in range(len(env) - duration):
        if np.all(env[i:i + duration+1] > THR_SIG):
            alarm[i] = np.max(
                env
            )  # If the current window of data surpasses the threshold, set an alarm.
            threshold = 0.1 * np.mean(
                env[i : i + duration+1]
            )  # Set a new threshold based on the mean of the current window.
            h += 1
        else:
            # Update noise
            if np.mean(env[i : i + duration+1]) < THR_SIG:
                noise = np.mean(
                    env[i : i + duration+1]
                )  # Update the noise value based on the mean of the current window.
            else:
                if len(noise_buf) > 0:
                    noise = np.mean(
                        noise_buf
                    )  # If available, use the mean of noise buffer to update the noise.
        thres_buf[i] = threshold  # Store the threshold value in the threshold buffer.
        noise_buf[i] = noise  # Store the noise value in the noise buffer.

            # Update threshold
        if h > 1:
            THR_SIG = noise + 0.50 * (
                    np.abs(threshold - noise)
                )  # Update the threshold using noise and threshold values.
        THR_buf[i] = THR_SIG  # Store the updated threshold value in the threshold buffer.

    if plot_results == 1:
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        plt.plot(input_signal)
        plt.plot(np.where(alarm != 0, np.max(input_signal), 0), "r", linewidth=2.5)
        plt.plot(THR_buf, "--g", linewidth=2.5)
        plt.title("Raw Signal and detected Onsets of activity")
        plt.legend(
            ["Raw Signal", "Detected Activity in Signal", "Adaptive Threshold"],
            loc="upper left",
        )
        plt.grid(True)
        plt.axis("tight")

        ax2 = plt.subplot(2, 1, 2)
        plt.plot(env)
        plt.plot(THR_buf, "--g", linewidth=2.5)
        plt.plot(thres_buf, "--r", linewidth=2)
        plt.plot(noise_buf, "--k", linewidth=2)
        plt.title("Smoothed Envelope of the signal (Hilbert Transform)")
        plt.legend(
            [
                "Smoothed Envelope of the signal (Hilbert Transform)",
                "Adaptive Threshold",
                "Activity level",
                "Noise Level",
            ]
        )
        plt.grid(True)
        plt.axis("tight")
        plt.tight_layout()
        plt.show()

    return alarm, env


def find_consecutive_groups(input_array):
    """_summary_
    Find consecutive groups of non-zero values in an input array.

    This function takes an input array `input_array`, converts it to a column vector, and identifies consecutive groups of
    non-zero values. It returns a 2D array where each row represents a group, with the first column containing
    the start index of the group and the second column containing the end index of the group.

    Args:
    input_array (ndarray): The input array.

    Returns:
    ind (ndarray): A 2D array where each row represents a group of consecutive non-zero values.
        The first column contains the start index of the group, and the second column contains the end index.
    """
    # Find indices of non-zeros elements
    temp = np.where(input_array)[0]

    # Find where the difference between indices is greater than 1
    idx = np.where(np.diff(temp) > 1)[0]

    # Initialize the output array
    ind = np.zeros((len(idx) + 1, 2), dtype=int)

    # Set the second column
    ind[:, 1] = temp[np.append(idx, -1)]

    # Set the first column
    ind[:, 0] = temp[np.insert(idx + 1, 0, 0)]

    return ind


def find_local_min_max(signal, threshold=None):
    """_summary_
    Find Local Minima and Maxima in a Given Signal.

    This function takes an input signal and identifies the indices of local minima and maxima.
    Optionally, a threshold can be provided to filter out minima and maxima that do not exceed the threshold.

    Args:
        signal (numpy.ndarray): The input signal.
        threshold (float or None, optional): Threshold for filtering out minima and maxima below and above this value, respectively.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray): A tuple containing two arrays:
            - minima_indices: Indices of local minima in the signal.
            - maxima_indices: Indices of local maxima in the signal.
    """
    # Find positive peaks in the signal
    maxima_indices, _ = scipy.signal.find_peaks(
        signal
    )

    # Find negative peaks in the inverted signal
    minima_indices, _ = scipy.signal.find_peaks(-signal)

    if threshold is not None:
        maxima_indices = maxima_indices[signal[maxima_indices] > threshold] + 1
        minima_indices = minima_indices[signal[minima_indices] < -threshold] + 1


    return minima_indices, maxima_indices


def identify_pulse_trains(signal):
    """_summary_
    Identify Pulse Trains in a Given Signal.

    This function takes an input signal and detects pulse trains within the signal.
    A pulse train is identified as a sequence of values with small intervals between adjacent values.

    Args:
        signal (numpy.ndarray): The input signal.

    Returns:
        list: A list of dictionaries, each containing information about a detected pulse train.
            Each dictionary has the following keys:
            - 'start': The index of the first value in the pulse train.
            - 'end': The index of the last value in the pulse train.
            - 'steps': The number of steps in the pulse train.
    """
    # Initialize an empty list to store detected pulse trains.
    pulse_trains = []

    # Initialize a flag to track whether we are within a pulse train.
    walking_flag = 0

    # Set an initial threshold value for pulse train detection.
    threshold = 3.5 * 40

    # Initialize a counter for the number of detected pulse trains.
    pulse_count = 0

    # Check if the signal has more than 2 elements.
    if len(signal) > 2:
        for i in range(len(signal) - 1):
            # Check if the difference between adjacent values is less than the threshold.
            if signal[i + 1] - signal[i] < threshold:
                if walking_flag == 0:
                    # If not already in a pulse train, start a new one.
                    pulse_trains.append({"start": signal[i], "steps": 1})
                    pulse_count += 1
                    walking_flag = 1
                else:
                    # If already in a pulse train, update the number of steps and threshold.
                    pulse_trains[pulse_count - 1]["steps"] += 1
                    threshold = (
                        1.5 * 40
                        + (signal[i] - pulse_trains[pulse_count - 1]["start"])
                        / pulse_trains[pulse_count - 1]["steps"]
                    )
            else:
                if walking_flag == 1:
                    # If leaving a pulse train, record its end and reset threshold.
                    pulse_trains[pulse_count - 1]["end"] = signal[i - 1]
                    walking_flag = 0
                    threshold = 3.5 * 40

    if walking_flag == 1:
        if signal[-1] - signal[-2] < threshold:
            # If still in a pulse train at the end, record its end and update steps.
            pulse_trains[-1]["end"] = signal[-1]
            pulse_trains[-1]["steps"] += 1
        else:
            # If leaving a pulse train at the end, record its end.
            pulse_trains[-1]["end"] = signal[-2]

    return pulse_trains


def convert_pulse_train_to_array(pulse_train_list):
    """_summary_
    Convert a List of Pulse Train Dictionaries to a 2D Array.

    This function takes a list of pulse train dictionaries and converts it into a 2D array.
    Each dictionary is expected to have keys 'start' and 'end', and the function creates an array
    where each row corresponds to a dictionary with the 'start' value in the first column and the
    'end' value in the second column.

    Args:
        pulse_train_list (list): A list of dictionaries containing pulse train information.

    Returns:
        numpy.ndarray: A 2D array where each row represents a pulse train with the 'start' value
            in the first column and the 'end' value in the second column.
    """
    # Initialize a 2D array with the same number of rows as pulse train dictionaries and 2 columns.
    array_representation = np.zeros((len(pulse_train_list), 2), dtype=np.uint64)

    # Iterate through the list of pulse train dictionaries.
    for i, pulse_train_dict in enumerate(pulse_train_list):
        # Iterate through the list of pulse train dictionaries.
        array_representation[i, 0] = pulse_train_dict["start"]

        # Iterate through the list of pulse train dictionaries.
        array_representation[i, 1] = pulse_train_dict["end"]

    return array_representation


def find_interval_intersection(set_a, set_b):
    """_summary_
    Find the Intersection of Two Sets of Intervals.

    Given two sets of intervals, this function computes their intersection and returns a new set
    of intervals representing the overlapping regions.

    Args:
        set_a (numpy.ndarray): The first set of intervals, where each row represents an interval with two values
            indicating the start and end points.
        set_b (numpy.ndarray): The second set of intervals, with the same structure as `set_a`.

    Returns:
        numpy.ndarray: A new set of intervals representing the intersection of intervals from `set_a` and `set_b`.
    """
    # Get the number of intervals in each set.
    num_intervals_a = set_a.shape[0]
    num_intervals_b = set_b.shape[0]

    # Initialize an empty list to store the intersection intervals.
    intersection_intervals = []

    # If either set of intervals is empty, return an empty array.
    if num_intervals_a == 0 or num_intervals_b == 0:
        return np.array(intersection_intervals)

    # Initialize indices and state variables for set_a and set_b traversal.
    index_a = 0
    index_b = 0
    state = 3

    # Traverse both sets of intervals and compute their intersection.
    while index_a < num_intervals_a and index_b < num_intervals_b:
        if state == 1:
            if set_a[index_a, 1] < set_b[index_b, 0]:
                index_a += 1
                state = 3
            elif set_a[index_a, 1] < set_b[index_b, 1]:
                intersection_intervals.append([set_b[index_b, 0], set_a[index_a, 1]])
                index_a += 1
                state = 2
            else:
                intersection_intervals.append(set_b[index_b, :])
                index_b += 1
        elif state == 2:
            if set_b[index_b, 1] < set_a[index_a, 0]:
                index_b += 1
                state = 3
            elif set_b[index_b, 1] < set_a[index_a, 1]:
                intersection_intervals.append([set_a[index_a, 0], set_b[index_b, 1]])
                index_b += 1
                state = 1
            else:
                intersection_intervals.append(set_a[index_a, :])
                index_a += 1
        elif state == 3:
            if set_a[index_a, 0] < set_b[index_b, 0]:
                state = 1
            else:
                state = 2

    return np.array(intersection_intervals)


def organize_and_pack_results(walking_periods, peak_steps):
    """Organize and Pack Walking Results with Associated Peak Steps.

    Given lists of walking periods and peak step indices, this function organizes and packs the results
    into a more structured format. It calculates the number of steps in each walking period, associates
    peak steps with their corresponding walking periods, and extends the duration of walking periods based
    on step time. The function also checks for overlapping walking periods and merges them.

    Args:
        walking_periods (list): List of tuples representing walking periods, where each tuple contains the start and end indices.
        peak_steps (list): List of peak step indices.

    Returns:
        tuple(list, list): A tuple containing two elements:
            - A list of dictionaries representing organized walking results, each dictionary contains:
                - 'start': Start index of the walking period.
                - 'end': End index of the walking period.
                - 'steps': Number of steps within the walking period.
                - 'mid_swing': List of peak step indices within the walking period.
            - A list of sorted peak step indices across all walking periods.
    """
    # Calculate the number of walking periods.
    num_periods = len(walking_periods)

    # Initialize a list of dictionaries to store organized walking results.
    organized_results = [
        {
            "start": walking_periods[i][0],
            "end": walking_periods[i][1],
            "steps": 0,
            "mid_swing": [],
        }
        for i in range(num_periods)
    ]

    # Initialize a list to store all peak step indices.
    all_mid_swing = []

    # Iterate through each walking period.
    for i in range(num_periods):
        # Find peak steps within the current walking period.
        steps_within_period = [
            p
            for p in peak_steps
            if organized_results[i]["start"] <= p <= organized_results[i]["end"]
        ]

        # Calculate the number of steps within the walking period.
        organized_results[i]["steps"] = len(steps_within_period)

        # Store the peak step indices within the walking period.
        organized_results[i]["mid_swing"] = steps_within_period

        # Add peak step indices to the list of all peak step indices.
        all_mid_swing.extend(steps_within_period)

        # Calculate step time based on detected peak steps
        if len(steps_within_period) > 2:
            step_time = sum(
                [
                    steps_within_period[j + 1] - steps_within_period[j]
                    for j in range(len(steps_within_period) - 1)
                ]
            ) / (len(steps_within_period) - 1)
            organized_results[i]["start"] = int(
                organized_results[i]["start"] - 1.5 * step_time / 2
            )
            organized_results[i]["end"] = int(
                organized_results[i]["end"] + 1.5 * step_time / 2
            )

    # Sort all peak step indices.
    all_mid_swing.sort()

    # Check for overlapping walking periods and merge them
    i = 0
    while i < num_periods - 1:
        if organized_results[i]["end"] >= organized_results[i + 1]["start"]:
            organized_results[i]["end"] = organized_results[i + 1]["end"]
            organized_results[i]["steps"] += organized_results[i + 1]["steps"]
            organized_results[i]["mid_swing"].extend(
                organized_results[i + 1]["mid_swing"]
            )
            organized_results.pop(i + 1)
            num_periods -= 1
        else:
            i += 1

    return organized_results, all_mid_swing


def max_peaks_between_zc(input_signal):
    """_summary_
    Find peaks and their locations from the vector input_signal between zero crossings.

    Args:
        input_signal (numpy.ndarray): Input column vector.

    Returns:
        pks (numpy.ndarray): Signed max/min values between zero crossings.
        ipks (numpy.ndarray): Locations of the peaks in the original vector.
    """
    # Check if the input is a valid column vector.
    if input_signal.shape[0] == 1:
        raise ValueError("X must be a column vector")
    if input_signal.size != len(input_signal):
        raise ValueError("X must be a column vector")

    # Flatten the input vector to ensure it's 1D.
    input_signal = input_signal.flatten()

    # Find the locations of zero crossings in the input vector.
    zero_crossings_locations = np.where(np.abs(np.diff(np.sign(input_signal))) == 2)[0] + 1 
    
    # Calculate the number of peaks.
    number_of_peaks = len(zero_crossings_locations) - 1

    def imax(input_signal):
        return np.argmax(input_signal)
    
    # Find the indices of the maximum values within each peak region.
    ipk = np.array([imax(np.abs(input_signal[zero_crossings_locations[i]:zero_crossings_locations[i + 1]])) for i in range(number_of_peaks)])
    ipks = zero_crossings_locations[:number_of_peaks] + ipk 
    ipks = ipks + 1

    # Retrieve the signed max/min values at the peak locations.
    pks = input_signal[ipks - 1]

    return pks, ipks


def signal_decomposition_algorithm(
    vertical_accelerarion_data, initial_sampling_frequency
):
    """_summary_
    Perform the Signal Decomposition algorithm on accelerometer data.

    Args:
        vertical_accelerarion_data (numpy.ndarray): Vertical Acceleration data.
        initial_sampling_frequency (float): Sampling frequency of the data.

    Returns:
        IC_seconds (numpy.ndarray): Detected IC (Initial Contact) timings in seconds.
        FC_seconds (numpy.ndarray): Detected FC (Foot-off Contact) timings in seconds.
    """
    # Define the target sampling frequency for processing.
    target_sampling_frequency = 40

    # Resample and interpolate the vertical acceleration data to the target sampling frequency.
    smoothed_vertical_accelerarion_data = resample_interpolate(
        vertical_accelerarion_data,
        initial_sampling_frequency,
        target_sampling_frequency,
    )

    # Load FIR filter designed and apply for the low SNR, impaired, asymmetric, and slow gait
    filtering_file = scipy.io.loadmat("ngmt/utils/FIR_2_3Hz_40.mat")
    num = filtering_file["Num"][0, :]
    width_of_pad = 10000 * len(num)

    smoothed_vertical_accelerarion_data_padded = np.pad(
        smoothed_vertical_accelerarion_data, width_of_pad, mode="wrap"
    )

    detrended_vertical_acceleration_signal = scipy.signal.filtfilt(
        num, 1, remove_40Hz_drift(smoothed_vertical_accelerarion_data_padded)
    )

    detrended_vertical_acceleration_signal_lpf_rmzp = (
        detrended_vertical_acceleration_signal[width_of_pad-1: len(detrended_vertical_acceleration_signal)-width_of_pad]
    )

    det_ver_acc_sig_LPInt = (
        scipy.integrate.cumulative_trapezoid(detrended_vertical_acceleration_signal_lpf_rmzp,initial='0')
        / target_sampling_frequency
    )

    # Perform a continuous wavelet transform on the siganl
    scales = 9
    wavelet = "gaus2"
    sampling_period = 1 / target_sampling_frequency
    coefficients, _ = pywt.cwt(
        det_ver_acc_sig_LPInt, np.arange(1, scales + 1), wavelet, sampling_period
    )
    desired_scale = 9
    smoothed_wavelet_result = coefficients[desired_scale - 1, :]
    smoothed_wavelet_result = smoothed_wavelet_result - np.mean(smoothed_wavelet_result)
    smoothed_wavelet_result = np.array(smoothed_wavelet_result)

    # Apply max_peaks_between_zc funtion to find peaks and their locations.
    pks1, ipks1 = max_peaks_between_zc(smoothed_wavelet_result.T)

    # Calculate indx1 (logical indices of negative elements)
    indx1 = pks1 < 0
    
    # Extract IC (indices of negative peaks)
    indices_of_negative_peaks = ipks1[indx1]

    # Convert IC to seconds
    IC_seconds = indices_of_negative_peaks / target_sampling_frequency

    # Apply continuous wavelet transform
    scales = 9
    wavelet = "gaus2"
    sampling_period = 1 / target_sampling_frequency
    coefficients, _ = pywt.cwt(
        smoothed_wavelet_result, np.arange(1, scales + 1), wavelet, sampling_period
    )
    desired_scale = 9
    accVLPIntCwt2 = coefficients[desired_scale - 1, :]
    accVLPIntCwt2 = accVLPIntCwt2 - np.mean(accVLPIntCwt2)
    accVLPIntCwt2 = np.array(accVLPIntCwt2)

    # Apply max_peaks_between_zc funtion to find peaks and their locations.
    pks2, ipks2 = max_peaks_between_zc(accVLPIntCwt2)

    # Calculate indx1 (logical indices of negative elements)
    indx2 = pks2 > 0

    # Extract IC (indices of negative peaks)
    final_contact = ipks2[indx2]

    # Convert IC to seconds
    FC_seconds = final_contact / target_sampling_frequency

    return IC_seconds, FC_seconds
