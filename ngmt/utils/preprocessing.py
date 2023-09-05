# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal


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
    recording_time = len(input_signal)
    x = np.arange(1, recording_time + 1)
    xq = np.arange(1, recording_time + 1, initial_sampling_rate / target_sampling_rate)
    interpolator = scipy.interpolate.interp1d(
        x, input_signal, kind="linear", axis=0, fill_value="extrapolate"
    )  # Create an interpolation function and apply it to the data
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
    numerator_coefficient = np.array(
        [1, -1]
    )  # The numerator coefficient vector of the filter.
    denominator_coefficient = np.array(
        [1, -0.9748]
    )  # The denominator coefficient vector of the filter.
    filtered_signal = scipy.signal.filtfilt(
        numerator_coefficient,
        denominator_coefficient,
        signal,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(numerator_coefficient), len(denominator_coefficient)) - 1),
    )

    return filtered_signal


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
    smooth_window (int): Window length for smoothing the envelope (default is 20).
    threshold_style (int): Threshold selection style: 0 for manual, 1 for automatic (default is 1).
    duration (int): Minimum duration of activity to be detected (default is 20).
    plot_results (int): Set to 1 for plotting results, 0 otherwise (default is 1).

    Returns:
    tuple(ndarray, ndarray): A tuple containing:
        - alarm (ndarray): Vector indicating active parts of the signal.
        - env (ndarray): Smoothed envelope of the signal.
    """
    # Input handling
    if len(locals()) < 5:  # If there is < 5 inputs.
        plot_results = 1  # Default value
        if len(locals()) < 4:  # If there is < 4 inputs.
            duration = 20  # Default value
            if len(locals()) < 3:  # If there is < 3 inputs.
                threshold_style = 1  # Default 1, means it is done automatically
                if len(locals()) < 2:  # If there is < 2 inputs.
                    smooth_window = 20  # Default value for smoothing length
                    if len(locals()) < 1:  # If there is < 1 inputs.
                        v = np.tile(
                            np.concatenate(
                                (0.1 * np.ones((200, 1)), np.ones((100, 1)))
                            ),
                            (10, 1),
                        )  # Generate true variance profile
                        input_signal = np.sqrt(v) * np.random.randn(*v.shape)

    # Calculate the analytical signal and get the envelope
    input_signal = (
        input_signal.flatten()
    )  # Return a copy of the preprocessed data into one dimension.
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
        hg = plt.figure()
        plt.plot(env)
        plt.title("Select a threshold on the graph")
        _, THR_SIG = plt.ginput(1)
        plt.close(hg)
    elif threshold_style == 1:
        THR_SIG = 4 * np.mean(env)

    # Set noise and signal levels
    noise = np.mean(env) * (
        1 / 3
    )  # Noise level: Set an initial estimate of the noise level
    threshold = np.mean(
        env
    )  # Signal level: It's used as a reference to distinguish between the background noise and the actual signal activity.

    # Initialize Buffers
    thres_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to a threshold.
    noise_buf = np.zeros(
        len(env) - duration
    )  # This buffer stores values related to the noise.
    THR_buf = np.zeros(len(env))  # This buffer stores threshold values.
    alarm = np.zeros_like(env)  # This buffer tracks alarm-related information.
    h = 1

    for i in range(len(env) - duration):
        if np.all(env[i : i + duration] > THR_SIG):
            alarm[i] = np.max(
                env
            )  # If the current window of data surpasses the threshold, set an alarm.
            threshold = 0.1 * np.mean(
                env[i : i + duration]
            )  # Set a new threshold based on the mean of the current window.
            h += 1
        else:
            # Update noise
            if np.mean(env[i : i + duration]) < THR_SIG:
                noise = np.mean(
                    env[i : i + duration]
                )  # Update the noise value based on the mean of the current window.
            else:
                if len(noise_buf) > 0:
                    noise = np.mean(
                        noise_buf
                    )  # If available, use the mean of noise buffer to update the noise.
                    thres_buf[
                        i
                    ] = threshold  # Store the threshold value in the threshold buffer.
                    noise_buf[i] = noise  # Store the noise value in the noise buffer.

            # Update threshold
            if h > 1:
                THR_SIG = noise + 0.50 * (
                    np.abs(threshold - noise)
                )  # Update the threshold using noise and threshold values.
                THR_buf[
                    i
                ] = THR_SIG  # Store the updated threshold value in the threshold buffer.

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
    temp = np.where(input_array)[0]  # find indices of non-zeros
    idx = np.where(np.diff(temp) > 1)[
        0
    ]  # find where the difference between indices is greater than 1
    ind = np.zeros((len(idx) + 1, 2), dtype=int)  # initialize the output array
    ind[:, 1] = temp[np.append(idx, -1)]  # set the second column
    ind[:, 0] = temp[np.insert(idx + 1, 0, 0)]  # set the first column

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
    signal_diff = np.diff(signal)
    zero_crossings = np.where(signal_diff[1:] * signal_diff[:-1] <= 0)[0]
    zero_crossings = zero_crossings + 1

    minima_indices = zero_crossings[signal_diff[zero_crossings] >= 0]
    maxima_indices = zero_crossings[signal_diff[zero_crossings] < 0]

    if threshold is not None:
        maxima_indices = maxima_indices[signal[maxima_indices] > threshold]
        minima_indices = minima_indices[signal[minima_indices] < -threshold]

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
    pulse_trains = []
    walking_flag = 0
    threshold = 3.5 * 40
    pulse_count = 0

    if len(signal) > 2:
        for i in range(len(signal) - 1):
            if signal[i + 1] - signal[i] < threshold:
                if walking_flag == 0:
                    pulse_trains.append({"start": signal[i], "steps": 1})
                    pulse_count += 1
                    walking_flag = 1
                else:
                    pulse_trains[pulse_count - 1]["steps"] += 1
                    threshold = (
                        1.5 * 40
                        + (signal[i] - pulse_trains[pulse_count - 1]["start"])
                        / pulse_trains[pulse_count - 1]["steps"]
                    )
            else:
                if walking_flag == 1:
                    pulse_trains[pulse_count - 1]["end"] = signal[i - 1]
                    walking_flag = 0
                    threshold = 3.5 * 40

    if walking_flag == 1:
        if signal[-1] - signal[-2] < threshold:
            pulse_trains[-1]["end"] = signal[-1]
            pulse_trains[-1]["steps"] += 1
        else:
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
    array_representation = np.zeros((len(pulse_train_list), 2), dtype=np.uint64)

    for i, pulse_train_dict in enumerate(pulse_train_list):
        array_representation[i, 0] = pulse_train_dict[
            "start"
        ]  # Access the 'start' key within the dictionary
        array_representation[i, 1] = pulse_train_dict[
            "end"
        ]  # Access the 'end' key within the dictionary

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
    num_intervals_a = set_a.shape[0]
    num_intervals_b = set_b.shape[0]

    intersection_intervals = []

    if num_intervals_a == 0 or num_intervals_b == 0:
        return np.array(intersection_intervals)

    index_a = 0
    index_b = 0
    state = 3

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
    num_periods = len(walking_periods)
    organized_results = [
        {
            "start": walking_periods[i][0],
            "end": walking_periods[i][1],
            "steps": 0,
            "mid_swing": [],
        }
        for i in range(num_periods)
    ]
    all_mid_swing = []

    for i in range(num_periods):
        steps_within_period = [
            p
            for p in peak_steps
            if organized_results[i]["start"] <= p <= organized_results[i]["end"]
        ]
        organized_results[i]["steps"] = len(steps_within_period)
        organized_results[i]["mid_swing"] = steps_within_period
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
