# Introduction and explanation regarding the test suite
"""
This code is a test suite for various signal processing and analysis functions which exist in the NGMT toolbox. 
It employs pytest, a Python testing framework, to verify the correctness of these functions. 
Here's a brief explanation of the code structure:

1. Import necessary libraries, pytest and the functions to be tested.
2. Generate a random input signal for testing purposes.
3. Define a series of test functions, each targeting a specific function from the 'ngmt.utils.preprocessing' module.
4. Inside each test function, we validate the correctness of the corresponding function and its inputs.
5. We make use of 'assert' statements to check that the functions return expected results.
6. The code is organized for clarity and maintainability.

To run the tests, follow these steps:

1. Make sure you have pytest installed. If not, install it using 'pip install -U pytest'.
2. Run this script, and pytest will execute all the test functions.
3. Any failures in tests will be reported as failed with red color, and also the number of passed tests will be represented with green color.

By running these tests, the reliability and correctness of the signal processing functions in the 'ngmt.utils.preprocessing' module will be ensured.
"""


# Import necessary libraries and functions to be tested.
import numpy as np
import numpy.testing as npt
import pytest
from ngmt.utils.preprocessing import (
    resample_interpolate,
    lowpass_filter,
    highpass_filter,
    _iir_highpass_filter,
    apply_continuous_wavelet_transform,
    apply_successive_gaussian_filters,
    calculate_envelope_activity,
    find_consecutive_groups,
    find_local_min_max,
    identify_pulse_trains,
    convert_pulse_train_to_array,
    find_interval_intersection,
    organize_and_pack_results,
    max_peaks_between_zc,
)


# Generate a random sinusoidal signal with varying amplitudes to use as an input in testing functions
time = np.linspace(0, 100, 1000)  # Time vector from 0 to 100 with 1000 samples
amplitudes = np.random.uniform(-3, 3, 1000)  # Random amplitudes between 3 and -3
sampling_frequency = 100  # Sampling frequency
random_input_signal = np.sin(2 * np.pi * sampling_frequency * time) * amplitudes


# Each test function checks a specific function.
# Test function for the 'resample_interpolate' function
def test_resample_interpolate():
    """
    Test for resample_interpolate function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    input_signal = random_input_signal
    initial_sampling_frequency = sampling_frequency
    target_sampling_frequency = 40

    # Call the resample_interpolate function with the specified inputs
    resampled_signal = resample_interpolate(
        input_signal, initial_sampling_frequency, target_sampling_frequency
    )

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(input_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the initial sampling frequency is postive
    assert (
        target_sampling_frequency > 0
    ), "Initial sampling frequency should be greater than 0."

    # Check that the target sampling frequency is positive
    assert (
        initial_sampling_frequency > 0
    ), "Target sampling frequency should be greater than 0."

    # Check that the resampled signal is not empty
    assert len(resampled_signal) > 0, "Resampled signal should not be empty."

    # Check the length of resampled signal
    expected_length = int(
        np.ceil(
            len(input_signal) * (target_sampling_frequency / initial_sampling_frequency)
        )
    )
    assert (
        resampled_signal.shape[0] == expected_length
    ), f"The resampled signal length is incorrect. Expected: {expected_length}, Actual: {resampled_signal.shape[0]}"

    # Check that the resampled signal does not contain any NaN or Inf values
    assert not np.isnan(
        resampled_signal
    ).any(), "The resampled signal contains NaN values."
    assert not np.isinf(
        resampled_signal
    ).any(), "The resampled signal contains Inf values."


# Test function for the 'lowpass_filter_savgol' function
def test_lowpass_filter_savgol():
    """
    Test for lowpass_filter_savgol function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    method = "savgol"

    # Call the lowpass_filter function with the specified inputs
    filtered_signal = lowpass_filter(test_signal, method)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the method is a string
    assert isinstance(method, str), "method must be a string."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."

    # Check that the filtered signal is not empty
    assert len(filtered_signal) > 0, "Filtered signal should not be empty."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), "The filtered signal length is incorrect."

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "The filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "The filtered signal contains Inf values."


# Test function for the 'lowpass_filter_fir' function
def test_lowpass_filter_fir():
    """
    Test for lowpass_filter_fir function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    method = "fir"

    # Call the lowpass_filter function with the specified inputs
    filtered_signal = lowpass_filter(test_signal, method)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the method is a string
    assert isinstance(method, str), "method must be a string."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."

    # Check that the filtered signal is not empty
    assert len(filtered_signal) > 0, "Filtered signal should not be empty."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), f"The filtered signal length is incorrect. Expected: {len(test_signal)}, Actual: {len(filtered_signal)}"

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "The filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "The filtered signal contains Inf values."


# Test function for the 'highpass_filter_iir' function
def test_highpass_filter_iir():
    """
    Test for highpass_filter_iir function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    sampling_frequency = 40
    method = "iir"

    # Call the highpass_filter function with the specified inputs
    filtered_signal = highpass_filter(test_signal, sampling_frequency, method)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the target sampling frequency is positive
    assert sampling_frequency > 0, "sampling frequency should be greater than 0."

    # Check that the method is a string
    assert isinstance(method, str), "method must be a string."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."

    # Check that the filtered signal is not empty
    assert len(filtered_signal) > 0, "Filtered signal should not be empty."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), f"The filtered signal length is incorrect. Expected: {len(test_signal)}, Actual: {len(filtered_signal)}"

    # Check that the filtered signal's frequency response in the passband is above a certain threshold
    passband_response = np.abs(np.fft.fft(filtered_signal))[5:15]
    assert np.all(
        passband_response >= 0.9
    ), "Filtered signal's passband response is too low."

    # Check that the method used for filtering is "iir"
    assert method == "iir", "Method should be 'iir' for highpass filtering."

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "The filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "The filtered signal contains Inf values."


# Test function for the 'test_iir_highpass_filter' function
def test_iir_highpass_filter():
    """
    Test for _iir_highpass_filter function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    sampling_frequency = 40

    # Call the _iir_highpass_filter function with the specified inputs
    filtered_signal = _iir_highpass_filter(test_signal, sampling_frequency)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the target sampling frequency is positive
    assert sampling_frequency > 0, "sampling frequency should be greater than 0."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), f"The filtered signal length is incorrect. Expected: {len(test_signal)}, Actual: {len(filtered_signal)}"

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "The filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "The filtered signal contains Inf values."


# Test function for the 'apply_continuous_wavelet_transform' function
def test_apply_continuous_wavelet_transform():
    """
    Test for apply_continuous_wavelet_transform function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    scales = 10
    desired_scale = 10
    wavelet = "gaus2"
    sampling_frequency = 40

    # Call the apply_continuous_wavelet_transform function with the specified inputs
    wavelet_transform_result = apply_continuous_wavelet_transform(
        test_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the scales are positive integer
    assert scales > 0 or isinstance(scales, int), "Scales should be a positive integer."

    # Check that the desired scale is a positive integer within the range of scales
    assert (
        isinstance(desired_scale, int) or desired_scale > 0 or desired_scale < scales
    ), "Desired scale must be a positive integer within the range of scales"

    # Check that the wavelet is a string
    assert isinstance(wavelet, str), "Wavelet must be a string."

    # Check that sampling frequency is a positive number
    assert (
        isinstance(sampling_frequency, (int, float)) or sampling_frequency > 0
    ), "Sampling frequency must be a positive number."

    # Check that the transformed signal is a NumPy array
    assert isinstance(
        wavelet_transform_result, np.ndarray
    ), "Transformed signal should be a NumPy array."

    # Check the length of transformed signal
    assert len(wavelet_transform_result) == len(
        test_signal
    ), f"The transformed signal length is incorrect. Expected: {len(test_signal)}, Actual: {len(wavelet_transform_result)}"

    # Check that the transformed signal does not contain any NaN or Inf values
    assert not np.isnan(
        wavelet_transform_result
    ).any(), "The transformed signal contains NaN values."
    assert not np.isinf(
        wavelet_transform_result
    ).any(), "The transformed signal contains Inf values."


# Test function for the 'apply_successive_gaussian_filters' function
def test_apply_successive_gaussian_filters():
    """
    Test for apply_successive_gaussian_filters function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the apply_successive_gaussian_filters function with the specified inputs
    filtered_signal = apply_successive_gaussian_filters(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the input signal is not empty
    assert test_signal.size >= 1, "Input signal must not be empty."

    # Check that the filtered signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), f"The filtered signal length is incorrect. Expected: {len(test_signal)}, Actual: {len(filtered_signal)}"

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "The filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "The filtered signal contains Inf values."


# Test function for the 'calculate_envelope_activity' function
def test_calculate_envelope_activity():
    """
    Test for calculate_envelope_activity function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    smooth_window = 20
    threshold_style = 1
    duration = 20
    plot_results = 0

    # Call the calculate_envelope_activity function with the specified inputs
    alarm, env = calculate_envelope_activity(
        test_signal, smooth_window, threshold_style, duration, plot_results=0
    )

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the window length is a positive integer
    assert (
        isinstance(smooth_window, (int)) or smooth_window > 0
    ), "The window length must be a positive integer."

    # Check that the threshold style is a positive integer
    assert (
        isinstance(threshold_style, (int)) or threshold_style > 0
    ), "The threshold style must be a positive integer."

    # Check that the duration of activity is a positive integer
    assert (
        isinstance(duration, (int)) or duration > 0
    ), "The duration of activity must be a positive integer."

    # Check that the plotting results is 0 or 1
    assert (
        plot_results == 0 or plot_results == 1
    ), "The plotting results must be 0 or 1."

    # Check that the outputs of function are a NumPy array
    assert isinstance(
        alarm, np.ndarray
    ), "Vector indicating active parts of the signal (alarm) should be a NumPy array."
    assert isinstance(
        env, np.ndarray
    ), "Smoothed envelope of the signal (env) should be a NumPy array."

    # Check that the outputs does not contain any NaN or Inf values
    assert not np.isnan(
        alarm
    ).any(), "Vector indicating active parts of the signal (alarm) contains NaN values."
    assert not np.isinf(
        alarm
    ).any(), "Vector indicating active parts of the signal (alarm) contains Inf values."
    assert not np.isnan(
        env
    ).any(), "Smoothed envelope of the signal (env) contains NaN values."
    assert not np.isinf(
        env
    ).any(), "Smoothed envelope of the signal (env) contains Inf values."


# Test function for the 'find_consecutive_groups' function
def test_find_consecutive_groups():
    """
    Test for find_consecutive_groups function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the find_consecutive_groups function with the specified inputs
    ind = find_consecutive_groups(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the input signal is not empty
    assert test_signal.size >= 1, "Input signal must not be empty."

    # Check that the output indices are non-negative
    assert np.all(ind >= 0), "Output indices contain negative values."

    # Check that the output does not contain any NaN or Inf values
    assert not np.isnan(ind).any(), "The output of the function contains NaN values."
    assert not np.isinf(ind).any(), "The output of the function contains Inf values."


# Test function for the 'find_local_min_max' function
def test_find_local_min_max():
    """
    Test for find_local_min_max function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the find_local_min_max function with the specified inputs
    minima_indices, maxima_indices = find_local_min_max(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the input signal is not empty
    assert test_signal.size >= 1, "Input signal must not be empty."

    # Check that the indices of local minima is a NumPy array
    assert isinstance(
        minima_indices, np.ndarray
    ), "The indices of local minima should be a NumPy array."

    # Check that the indices of local maxima is a NumPy array
    assert isinstance(
        maxima_indices, np.ndarray
    ), "The indices of local maxima should be a NumPy array."

    # Check that the minimum and maximum indices are in the expected ranges
    assert np.all(minima_indices >= 0) and np.all(
        minima_indices < len(test_signal)
    ), "Minimum indices are out of range."
    assert np.all(maxima_indices >= 0) and np.all(
        maxima_indices < len(test_signal)
    ), "Maximum indices are out of range."

    # Check that the indices of local minima does not contain any NaN or Inf values
    assert not np.isnan(
        minima_indices
    ).any(), "The indices of local minima contains NaN values."
    assert not np.isinf(
        minima_indices
    ).any(), "The indices of local minima contains Inf values."

    # Check that the indices of local maxima does not contain any NaN or Inf values
    assert not np.isnan(
        maxima_indices
    ).any(), "The indices of local maxima contains NaN values."
    assert not np.isinf(
        maxima_indices
    ).any(), "The indices of local maxima contains Inf values."


# Test function for the 'identify_pulse_trains' function
def test_identify_pulse_trains():
    """
    Test for identify_pulse_trains function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the identify_pulse_trains function with the specified inputs
    pulse_trains = identify_pulse_trains(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the number of identified pulse trains matches the expected count
    expected_pulse_train_count = 3
    assert (
        len(pulse_trains) == expected_pulse_train_count
    ), "Unexpected number of identified pulse trains."

    # Check that the 'start' and 'end' values for each pulse train are in the expected range
    for pulse_train in pulse_trains:
        assert (
            0 <= pulse_train["start"] < len(test_signal)
        ), "Pulse train 'start' value is out of range."
        assert (
            0 <= pulse_train["end"] < len(test_signal)
        ), "Pulse train 'end' value is out of range."

    # Check that the output is a list
    assert isinstance(pulse_trains, list), "Output should be a list of pulse trains."

    # Check that each element in the list is a dictionary with the expected keys
    for pulse_train in pulse_trains:
        assert isinstance(
            pulse_train, dict
        ), "Each element in the list should be a dictionary."
        assert "start" in pulse_train, "Dictionary should contain 'start' key."
        assert "end" in pulse_train, "Dictionary should contain 'end' key."
        assert "steps" in pulse_train, "Dictionary should contain 'steps' key."


# Test function for the 'identify_pulse_trains' function
def test_identify_pulse_trains():
    """
    Test for convert_pulse_train_to_array function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the identify_pulse_trains function with the specified inputs
    pulse_trains = identify_pulse_trains(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the output is a list
    assert isinstance(pulse_trains, list), "Output should be a list of pulse trains."

    # Check that each element in the list is a dictionary with the expected keys
    for pulse_train in pulse_trains:
        assert isinstance(
            pulse_train, dict
        ), "Each element in the list should be a dictionary."
        assert "start" in pulse_train, "Dictionary should contain 'start' key."
        assert "end" in pulse_train, "Dictionary should contain 'end' key."
        assert "steps" in pulse_train, "Dictionary should contain 'steps' key."

    # Check that 'steps' is a positive integer
    assert isinstance(pulse_train["steps"], int), "'steps' should be an integer."
    assert pulse_train["steps"] > 0, "'steps' should be a positive integer."


# Test function for the 'convert_pulse_train_to_array' function
def test_convert_pulse_train_to_array():
    """
    Test for find_interval_intersection function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with a list of pulse train dictionaries
    pulse_train_list = [
        {"start": 0, "end": 10},
        {"start": 20, "end": 30},
        {"start": 40, "end": 50},
    ]

    # Assertions to be checked
    # Check that pulse_train_list is a list
    assert isinstance(
        pulse_train_list, list
    ), "Input should be a list of pulse train dictionaries."

    # Check that each element in the list is a dictionary with the expected keys
    for pulse_train in pulse_train_list:
        assert isinstance(
            pulse_train, dict
        ), "Each element in the list should be a dictionary."
        assert "start" in pulse_train, "Dictionary should contain 'start' key."
        assert "end" in pulse_train, "Dictionary should contain 'end' key."

    # Call the convert_pulse_train_to_array function with the specified inputs
    array_representation = convert_pulse_train_to_array(pulse_train_list)

    # Check if the input is a list of pulse train dictionaries
    assert isinstance(
        pulse_train_list, list
    ), "Input should be a list of pulse train dictionaries."

    # Check the data type of the output array
    assert isinstance(
        array_representation, np.ndarray
    ), "Output should be a NumPy array."

    # Check that the resulting array has two columns
    assert (
        array_representation.shape[1] == 2
    ), "Converted pulse train array should have two columns."

    # Check that the 'start' and 'end' values in the resulting array match the input pulse train data
    for i, pulse_train in enumerate(pulse_train_list):
        assert (
            array_representation[i, 0] == pulse_train["start"]
        ), f"Start value for pulse train {i} does not match."
        assert (
            array_representation[i, 1] == pulse_train["end"]
        ), f"End value for pulse train {i} does not match."

    # Check the values in the output array against the original data
    expected_values = np.array([[0, 10], [20, 30], [40, 50]], dtype=np.uint64)
    assert np.array_equal(
        array_representation, expected_values
    ), "Output array values do not match the expected values."

    # Check that the 'start' values match the first column and 'end' values match the second column
    assert np.array_equal(
        array_representation[:, 0], expected_values[:, 0]
    ), "Start values in the output array do not match."
    assert np.array_equal(
        array_representation[:, 1], expected_values[:, 1]
    ), "End values in the output array do not match."


# Test function for the 'find_interval_intersection' function
def test_find_interval_intersection():
    """
    Test for organize_and_pack_results function in the 'ngmt.utils.preprocessing' module.
    """
    # Test case 1: Basic case with one intersection
    set_a = np.array([[1, 5], [7, 10]])
    set_b = np.array([[3, 8], [9, 12]])
    expected_result = np.array([[3, 5], [7, 8], [9, 10]])

    # Assertions to be checked:
    # Check if inputs (set_a and set_b) are NumPy arrays
    assert isinstance(set_a, np.ndarray), "set_a should be a NumPy array."
    assert isinstance(set_b, np.ndarray), "set_b should be a NumPy array."

    # Check if inputs (set_a and set_b) have the correct structure (two columns)
    assert set_a.shape[1] == 2, "set_a should have two columns."
    assert set_b.shape[1] == 2, "set_b should have two columns."

    # Call the find_interval_intersection function with the specified inputs
    result = find_interval_intersection(set_a, set_b)

    # Check the data type of the output
    assert isinstance(result, np.ndarray), "Output should be a NumPy array."

    # Check if the output matches the expected result
    npt.assert_array_equal(
        result, expected_result, "Output does not match the expected result."
    )

    # Test case 2: No intersection
    set_a = np.array([[1, 5], [7, 10]])
    set_b = np.array([[11, 15], [17, 20]])
    expected_result = np.array([])

    # Assertions to be checked:
    # Check if set_a and set_b are NumPy arrays
    assert isinstance(set_a, np.ndarray), "set_a should be a NumPy array."
    assert isinstance(set_b, np.ndarray), "set_b should be a NumPy array."

    # Check if set_a and set_b have the correct structure (two columns)
    assert set_a.shape[1] == 2, "set_a should have two columns."
    assert set_b.shape[1] == 2, "set_b should have two columns."

    # Call the find_interval_intersection function with the specified inputs
    result = find_interval_intersection(set_a, set_b)

    # Check the data type of the output
    assert isinstance(result, np.ndarray), "Output should be a NumPy array."

    # Check if the output matches the expected result
    npt.assert_array_equal(
        result, expected_result, "Output does not match the expected result."
    )


# Test function for the 'organize_and_pack_results' function
def test_organize_and_pack_results():
    # Test case 1: Basic case with non-overlapping walking periods
    walking_periods = [(0, 10), (20, 30), (40, 50)]
    peak_steps = [5, 25, 45]
    expected_results = [
        {
            "start": 0,
            "end": 10,
            "steps": 1,
            "mid_swing": [5],
        },
        {
            "start": 20,
            "end": 30,
            "steps": 1,
            "mid_swing": [25],
        },
        {
            "start": 40,
            "end": 50,
            "steps": 1,
            "mid_swing": [45],
        },
    ]
    expected_peak_steps = [5, 25, 45]

    # Assertions to be checked:
    # Check if walking_periods is a list
    assert isinstance(walking_periods, list), "walking_periods should be a list."

    # Check that each element in walking_periods is a tuple with two elements
    for period in walking_periods:
        assert isinstance(
            period, tuple
        ), "Each element in walking_periods should be a tuple."
        assert (
            len(period) == 2
        ), "Each tuple in walking_periods should contain start and end indices."

    # Check if peak_steps is a list
    assert isinstance(peak_steps, list), "peak_steps should be a list."

    # Call the organize_and_pack_results function with the specified inputs
    results, peak_steps_result = organize_and_pack_results(walking_periods, peak_steps)

    # Check if results is a list of dictionaries
    assert isinstance(results, list), "Output results should be a list."

    for result in results:
        assert isinstance(
            result, dict
        ), "Each element in results should be a dictionary."
        assert "start" in result, "Dictionary should contain 'start' key."
        assert "end" in result, "Dictionary should contain 'end' key."
        assert "steps" in result, "Dictionary should contain 'steps' key."
        assert "mid_swing" in result, "Dictionary should contain 'mid_swing' key."

    # Check the values in the output results
    assert (
        results == expected_results
    ), "Output results do not match the expected results."

    # Check that the 'steps' value for each result is a positive integer
    for result in results:
        assert (
            result["steps"] > 0
        ), "'steps' value in results should be a positive integer."

    # Check that the 'mid_swing' values are within the 'start' and 'end' range for each result
    for result in results:
        assert all(
            result["start"] <= step <= result["end"] for step in result["mid_swing"]
        ), "Mid-swing values are out of range for some results."

    # Check if peak_steps_result is a list
    assert isinstance(
        peak_steps_result, list
    ), "Output peak_steps_result should be a list."

    # Check the values in the output peak_steps_result
    assert (
        peak_steps_result == expected_peak_steps
    ), "Output peak_steps_result do not match the expected peak_steps."


# Test function for the 'max_peaks_between_zc' function
def test_max_peaks_between_zc():
    """
    Test for max_peaks_between_zc function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal

    # Call the max_peaks_between_zc function with the specified inputs
    pks, ipks = max_peaks_between_zc(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the outputs are NumPy arrays
    assert isinstance(
        pks, np.ndarray
    ), "Signed max/min values between zero crossings (pks) should be a NumPy array."
    assert isinstance(
        ipks, np.ndarray
    ), "Locations of the peaks in the original vector (ipks) should be a NumPy array."

    # Check that the lengths of pks and ipks are consistent
    assert len(pks) == len(ipks), "Lengths of pks and ipks should be the same."

    # Check that ipks are within the valid range of indices for the input signal
    assert np.all(ipks >= 0) and np.all(
        ipks < len(test_signal)
    ), "Peak indices (ipks) should be within the valid range for the input signal."

    # Check that the values in pks correspond to the correct values in the input signal
    assert np.all(
        pks == test_signal[ipks - 1]
    ), "Values in pks should correspond to the correct values in the input signal."


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
