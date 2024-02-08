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
import pandas as pd
import numpy as np
import warnings
import numpy.testing as npt
import pytest
import scipy
import matplotlib.pyplot as plt
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection
from ngmt.modules.pam import PhysicalActivityMonitoring
from matplotlib.testing.compare import compare_images
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
    signal_decomposition_algorithm,
    classify_physical_activity,
)
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection

# Generate a random sinusoidal signal with varying amplitudes to use as an input in testing functions
time = np.linspace(0, 100, 1000)  # Time vector from 0 to 100 with 1000 samples
amplitudes = np.random.uniform(-3, 3, 1000)  # Random amplitudes between 3 and -3
sampling_frequency = 100  # Sampling frequency
random_input_signal = np.sin(2 * np.pi * sampling_frequency * time) * amplitudes


# Each test function checks a specific function.
# Test function for the 'resample_interpolate' function: Case 1
def test_resample_interpolate():
    """
    Test for resample_interpolate function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    input_signal = random_input_signal
    initial_sampling_frequency = sampling_frequency
    target_sampling_frequency = 40

    # Call the resample_interpolate function with the specified inputs
    with warnings.catch_warnings():
        # Ignore the specific warning (invalid value encountered in divide)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        resampled_signal = resample_interpolate(
            input_signal, initial_sampling_frequency, target_sampling_frequency
        )


# Test function for the 'resample_interpolate' function: Case 2
def test_resample_interpolate_non_numpy_input():
    # Test with non-NumPy array input
    input_signal = [1, 2, 3, 4, 5]
    initial_sampling_frequency = 1
    target_sampling_frequency = 0.5

    with pytest.raises(ValueError, match="NumPy array"):
        resample_interpolate(
            input_signal, initial_sampling_frequency, target_sampling_frequency
        )


# Other test cases for the resample_interpolate function
def test_resample_interpolate():
    # Test with valid inputs
    input_signal = np.random.rand(100)  # Sample input signal
    resampled_signal = resample_interpolate(
        input_signal, initial_sampling_frequency=100, target_sampling_frequency=40
    )
    assert len(resampled_signal) > 0  # Check if resampled signal is not empty

    # Test with initial sampling frequency not a positive float
    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=0, target_sampling_frequency=40
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=-100, target_sampling_frequency=40
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=0
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=-40
        )

    # Test with target sampling frequency not a positive float
    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=0
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=-40
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=0
        )

    with pytest.raises(ValueError):
        resample_interpolate(
            input_signal, initial_sampling_frequency=100, target_sampling_frequency=-40
        )


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


# Test function for the 'lowpass_filter' function: case 1
def test_lowpass_filter_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]
    method = "savgol"

    with pytest.raises(ValueError, match="Input data must be a numpy.ndarray"):
        lowpass_filter(input_signal, method=method)


# Test function for the 'lowpass_filter' function: case 2
def test_lowpass_filter_invalid_method():
    # Test with invalid filter method
    input_signal = np.array([1, 2, 3, 4, 5])
    method = "invalid_method"

    with pytest.raises(ValueError, match="Invalid filter method specified"):
        lowpass_filter(input_signal, method=method)


# Test function for the 'lowpass_filter' function: case 3
def test_lowpass_filter_butter_no_order():
    # Test Butterworth filter without specifying order
    input_signal = np.array([1, 2, 3, 4, 5])
    method = "butter"
    cutoff_freq_hz = 2.0
    sampling_rate_hz = 10.0

    with pytest.raises(
        ValueError, match="For Butterworth filter, 'order' must be specified."
    ):
        lowpass_filter(
            input_signal,
            method=method,
            cutoff_freq_hz=cutoff_freq_hz,
            sampling_rate_hz=sampling_rate_hz,
        )


# Other teest cases for the lowpass_filter function
def test_lowpass_filter():
    # Test with valid inputs
    input_signal = np.random.rand(100)  # Sample input signal

    # Test with invalid method (not a string)
    with pytest.raises(ValueError):
        lowpass_filter(input_signal, method=123)

    # Test with invalid method (empty string)
    with pytest.raises(ValueError):
        lowpass_filter(input_signal, method="")

    # Test with invalid method (nonexistent method)
    with pytest.raises(ValueError):
        lowpass_filter(input_signal, method="invalid_method")


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


# Additional test cases for Savitzky-Golay filter
def test_lowpass_filter_savgol_specific():
    """
    Test specific parameters for Savitzky-Golay filter in the 'ngmt.utils.preprocessing' module.
    """
    # Test with specific parameters
    test_signal = np.ones(100)
    window_length = 5
    polynomial_order = 2

    # Call the lowpass_filter function with specific parameters
    filtered_signal = lowpass_filter(
        test_signal,
        method="savgol",
        window_length=window_length,
        polynomial_order=polynomial_order,
    )

    # Assertions to be checked:
    assert len(filtered_signal) == len(
        test_signal
    ), "Filtered signal length is incorrect."


# Additional test cases for Butterworth filter
def test_lowpass_filter_butter_specific():
    """
    Test specific parameters for Butterworth filter in the 'ngmt.utils.preprocessing' module.
    """
    # Test with specific parameters
    test_signal = np.ones(100)
    order = 3
    cutoff_freq_hz = 10
    sampling_rate_hz = 100

    # Call the lowpass_filter function with specific parameters
    filtered_signal = lowpass_filter(
        test_signal,
        method="butter",
        order=order,
        cutoff_freq_hz=cutoff_freq_hz,
        sampling_rate_hz=sampling_rate_hz,
    )

    # Assertions to be checked:
    assert len(filtered_signal) == len(
        test_signal
    ), "Filtered signal length is incorrect."


# Test function for the 'highpass_filter_iir' function
def test_highpass_filter_iir():
    """Test for highpass_filter_iir function in the 'ngmt.utils.preprocessing' module."""
    # Test with inputs
    test_signal = np.random.rand(100)
    sampling_frequency = 40
    method = "iir"

    # Call the highpass_filter function with the specified inputs
    filtered_signal = highpass_filter(test_signal, sampling_frequency, method)

    # Assertions
    # Check that the input signal is a NumPy array
    assert isinstance(
        test_signal, np.ndarray
    ), "Assertion Error: Input signal should be a NumPy array."

    # Check that the target sampling frequency is positive
    assert (
        sampling_frequency > 0
    ), "Assertion Error: Sampling frequency should be greater than 0."

    # Check that the method is a string
    assert isinstance(method, str), "Assertion Error: Method must be a string."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Assertion Error: Filtered signal should be a NumPy array."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), "Assertion Error: Filtered signal length is incorrect."

    # Check that the method used for filtering is "iir"
    assert (
        method == "iir"
    ), "Assertion Error: Method should be 'iir' for highpass filtering."

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "Assertion Error: Filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "Assertion Error: Filtered signal contains Inf values."


# Test function for the '_iir_highpass_filter' function: case 1
def test_iir_highpass_filter_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]
    sampling_frequency = 40

    with pytest.raises(
        ValueError,
        match="Invalid input data. The 'signal' must be a NumPy array, and 'sampling_frequency' must be a positive number.",
    ):
        _iir_highpass_filter(input_signal, sampling_frequency)


# Test function for the '_iir_highpass_filter' function: case 2
def test_iir_highpass_filter():
    """Test for _iir_highpass_filter function in the 'ngmt.utils.preprocessing' module."""
    # Test with inputs
    test_signal = np.random.rand(100)
    sampling_frequency = 40

    # Call the _iir_highpass_filter function with the specified inputs
    filtered_signal = _iir_highpass_filter(test_signal, sampling_frequency)

    # Assertions
    # Check that the input signal is a NumPy array
    assert isinstance(
        test_signal, np.ndarray
    ), "Assertion Error: Input signal should be a NumPy array."

    # Check that the target sampling frequency is positive
    assert (
        sampling_frequency > 0
    ), "Assertion Error: Sampling frequency should be greater than 0."

    # Check that the filtered_signal is a NumPy array
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Assertion Error: Filtered signal should be a NumPy array."

    # Check the length of filtered signal
    assert len(filtered_signal) == len(
        test_signal
    ), "Assertion Error: Filtered signal length is incorrect."

    # Check that the filtered signal does not contain any NaN or Inf values
    assert not np.isnan(
        filtered_signal
    ).any(), "Assertion Error: Filtered signal contains NaN values."
    assert not np.isinf(
        filtered_signal
    ).any(), "Assertion Error: Filtered signal contains Inf values."


# Test function for the 'highpass_filter' function
def test_highpass_filter_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]
    sampling_frequency = 40
    method = "iir"

    with pytest.raises(
        ValueError,
        match="Invalid input data. The 'signal' must be a NumPy array, and 'sampling_frequency' must be a positive number.",
    ):
        highpass_filter(input_signal, sampling_frequency, method=method)


# Other test cases for the highpass_filter function
def test_highpass_filter():
    # Test with valid inputs
    input_signal = np.random.rand(100)  # Sample input signal

    # Test with invalid method (not a string)
    with pytest.raises(ValueError):
        highpass_filter(input_signal, method=123)

    # Test with invalid method (empty string)
    with pytest.raises(ValueError):
        highpass_filter(input_signal, method="")

    # Test with invalid method (nonexistent method)
    with pytest.raises(ValueError):
        highpass_filter(input_signal, method="invalid_method")

    # Test with valid method
    filtered_signal = highpass_filter(input_signal, method="iir")
    assert isinstance(
        filtered_signal, np.ndarray
    ), "Filtered signal should be a NumPy array."


# Test function for the 'apply_continuous_wavelet_transform' function: case 1
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
    assert (
        isinstance(scales, int) and scales > 0
    ), "Scales should be a positive integer."

    # Check that the desired scale is a positive integer within the range of scales
    assert (
        isinstance(desired_scale, int) and 0 < desired_scale <= scales
    ), "Desired scale must be a positive integer within the range of scales."

    # Check that the wavelet is a string
    assert isinstance(wavelet, str), "Wavelet must be a string."

    # Check that sampling frequency is a positive number
    assert (
        isinstance(sampling_frequency, (int, float)) and sampling_frequency > 0
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

    # Additional test case for minimum valid values of scales and desired_scale
    min_scales = 1
    min_desired_scale = 1
    wavelet_transform_result_min = apply_continuous_wavelet_transform(
        test_signal, min_scales, min_desired_scale, wavelet, sampling_frequency
    )
    assert isinstance(
        wavelet_transform_result_min, np.ndarray
    ), "Transformed signal for minimum scales and desired_scale should be a NumPy array."
    assert len(wavelet_transform_result_min) == len(
        test_signal
    ), "The transformed signal length for minimum scales and desired_scale is incorrect."

    # Test case for negative scales or desired_scale
    negative_scales = -5
    negative_desired_scale = -2
    wavelet_transform_result_negative = apply_continuous_wavelet_transform(
        test_signal,
        negative_scales,
        negative_desired_scale,
        wavelet,
        sampling_frequency,
    )
    assert (
        wavelet_transform_result_negative is None
    ), "Function should handle negative scales or desired_scale and return None."

    # Test case for non-integer scales or desired_scale
    non_integer_scales = 5.5
    non_integer_desired_scale = 2.3
    wavelet_transform_result_non_integer = apply_continuous_wavelet_transform(
        test_signal,
        non_integer_scales,
        non_integer_desired_scale,
        wavelet,
        sampling_frequency,
    )
    assert (
        wavelet_transform_result_non_integer is None
    ), "Function should handle non-integer scales or desired_scale and return None."


# Test function for the 'apply_continuous_wavelet_transform' function: case 2
def test_apply_continuous_wavelet_transform_valid_input():
    # Test with valid input
    input_signal = np.array([1, 2, 3, 4, 5])
    scales = 10
    desired_scale = 5
    wavelet = "gaus2"
    sampling_frequency = 40

    result = apply_continuous_wavelet_transform(
        input_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Check that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Check that the length of the result is correct
    expected_length = len(input_signal)
    assert len(result) == expected_length

    # Check that the result does not contain any NaN or Inf values
    assert not np.isnan(result).any(), "The result contains NaN values."
    assert not np.isinf(result).any(), "The result contains Inf values."


# Test function for the 'apply_continuous_wavelet_transform' function: case 3
def test_apply_continuous_wavelet_transform_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]
    scales = 10
    desired_scale = 5
    wavelet = "gaus2"
    sampling_frequency = 40

    result = apply_continuous_wavelet_transform(
        input_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Check that the result is None due to invalid input
    assert result is None


# Test function for the 'apply_continuous_wavelet_transform' function: case 4
def test_apply_continuous_wavelet_transform_exception_handling():
    # Test exception handling
    input_signal = np.array([1, 2, 3, 4, 5])
    scales = "invalid"
    desired_scale = 5
    wavelet = "gaus2"
    sampling_frequency = 40

    result = apply_continuous_wavelet_transform(
        input_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Check that the result is None due to the exception
    assert result is None


# Test function for the 'apply_continuous_wavelet_transform' function: case 5
def test_apply_continuous_wavelet_transform_large_input():
    # Test with a large input signal
    input_signal = np.random.rand(10000)
    scales = 20
    desired_scale = 10
    wavelet = "gaus2"
    sampling_frequency = 40

    result = apply_continuous_wavelet_transform(
        input_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Check that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Check that the length of the result is correct
    expected_length = len(input_signal)
    assert len(result) == expected_length

    # Check that the result does not contain any NaN or Inf values
    assert not np.isnan(result).any(), "The result contains NaN values."
    assert not np.isinf(result).any(), "The result contains Inf values."


# Test function for the 'apply_continuous_wavelet_transform' function: case 6
def test_apply_continuous_wavelet_transform_zero_sampling_frequency():
    # Test with zero sampling frequency
    input_signal = np.array([1, 2, 3, 4, 5])
    scales = 10
    desired_scale = 5
    wavelet = "gaus2"
    sampling_frequency = 0

    result = apply_continuous_wavelet_transform(
        input_signal, scales, desired_scale, wavelet, sampling_frequency
    )

    # Check that the result is None due to zero sampling frequency
    assert result is None


# Test function for the 'apply_successive_gaussian_filters' function: case 1
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

    # Check that the filtered signal is either a NumPy array or None
    assert (
        isinstance(filtered_signal, np.ndarray) or filtered_signal is None
    ), "Filtered signal should be a NumPy array or None."

    # If filtered_signal is not None, check the length of filtered signal
    if filtered_signal is not None:
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

    # Additional test case for minimum signal length
    min_length_signal = np.array([5.0])
    filtered_min_length_signal = apply_successive_gaussian_filters(min_length_signal)
    assert isinstance(
        filtered_min_length_signal, np.ndarray
    ), "Filtered signal for minimum length signal should be a NumPy array."

    # Test case for a signal with a single data point
    single_point_signal = np.array([10.0])
    filtered_single_point_signal = apply_successive_gaussian_filters(
        single_point_signal
    )
    assert isinstance(
        filtered_single_point_signal, np.ndarray
    ), "Filtered signal for a single data point signal should be a NumPy array."

    # Test case for negative sigma
    negative_sigma_signal = random_input_signal
    negative_sigma_signal[0] = -2
    filtered_negative_sigma_signal = apply_successive_gaussian_filters(
        negative_sigma_signal
    )
    assert filtered_negative_sigma_signal is None or isinstance(
        filtered_negative_sigma_signal, np.ndarray
    ), "Function should handle negative sigma and return None or NumPy array."


# Test function for the 'apply_successive_gaussian_filters' function: case 2
def test_apply_successive_gaussian_filters_valid_input():
    # Test with valid input
    input_data = np.array([1, 2, 3, 4, 5])

    result = apply_successive_gaussian_filters(input_data)

    # Check that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Check that the length of the result is correct
    expected_length = len(input_data)
    assert len(result) == expected_length

    # Check that the result does not contain any NaN or Inf values
    assert not np.isnan(result).any(), "The result contains NaN values."
    assert not np.isinf(result).any(), "The result contains Inf values."


# Test function for the 'apply_successive_gaussian_filters' function: case 3
def test_apply_successive_gaussian_filters_invalid_input():
    # Test with invalid input data type
    input_data = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="NumPy array"):
        apply_successive_gaussian_filters(input_data)


# Test function for the 'apply_successive_gaussian_filters' function: case 4
def test_apply_successive_gaussian_filters_empty_input():
    # Test with empty input data
    input_data = np.array([])

    with pytest.raises(ValueError, match="Input data must not be empty."):
        apply_successive_gaussian_filters(input_data)


# Test function for the 'apply_successive_gaussian_filters' function: case 5
def test_apply_successive_gaussian_filters_large_input():
    # Test with a large input data
    input_data = np.random.rand(10000)

    result = apply_successive_gaussian_filters(input_data)

    # Check that the output is a NumPy array
    assert isinstance(result, np.ndarray)

    # Check that the length of the result is correct
    expected_length = len(input_data)
    assert len(result) == expected_length

    # Check that the result does not contain any NaN or Inf values
    assert not np.isnan(result).any(), "The result contains NaN values."
    assert not np.isinf(result).any(), "The result contains Inf values."


# Test function for the 'calculate_envelope_activity' function: case 1
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
        isinstance(smooth_window, int) and smooth_window > 0
    ), "The window length must be a positive integer."

    # Check that the threshold style is a positive integer
    assert (
        isinstance(threshold_style, int) and threshold_style > 0
    ), "The threshold style must be a positive integer."

    # Check that the duration of activity is a positive integer
    assert (
        isinstance(duration, int) and duration > 0
    ), "The duration of activity must be a positive integer."

    # Check that the plotting results is 0 or 1
    assert plot_results in [0, 1], "The plotting results must be 0 or 1."

    # Check that the outputs of the function are NumPy arrays
    assert isinstance(
        alarm, np.ndarray
    ), "Vector indicating active parts of the signal (alarm) should be a NumPy array."
    assert isinstance(
        env, np.ndarray
    ), "Smoothed envelope of the signal (env) should be a NumPy array."

    # Check that the outputs do not contain any NaN or Inf values
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

    # Additional assertions for specific scenarios
    assert len(alarm) > len(
        test_signal
    ), "Length of alarm vector should be greater than the length of the input signal."

    # Check that the length of the smoothed envelope vector is greater than or equal to the length of the input signal
    assert len(env) >= len(
        test_signal
    ), "Length of smoothed envelope vector should be greater than or equal to the length of the input signal."

    # Use pytest.approx for floating-point comparisons
    assert alarm[0] == pytest.approx(
        0.0
    ), "First element of alarm vector should be approximately 0.0."


# Test function for the 'calculate_envelope_activity' function: case 2
def test_calculate_envelope_activity_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="Input signal should be a NumPy array."):
        calculate_envelope_activity(input_signal)


# Test function for the 'calculate_envelope_activity' function: case 3
def test_calculate_envelope_activity_invalid_window_length():
    # Test with invalid smooth_window value
    input_signal = np.array([1, 2, 3, 4, 5])

    with pytest.raises(
        ValueError, match="The window length must be a positive integer."
    ):
        calculate_envelope_activity(input_signal, smooth_window=-20)


# Test function for the 'calculate_envelope_activity' function: case 4
def test_calculate_envelope_activity_invalid_threshold_style():
    # Test with invalid threshold_style value
    input_signal = np.array([1, 2, 3, 4, 5])

    with pytest.raises(
        ValueError, match="The threshold style must be a positive integer."
    ):
        calculate_envelope_activity(input_signal, threshold_style=-1)


# Test function for the 'calculate_envelope_activity' function: case 5
def test_calculate_envelope_activity_invalid_duration():
    # Test with invalid duration value
    input_signal = np.array([1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="The duration must be a positive integer."):
        calculate_envelope_activity(input_signal, duration=0)


# Test function for the 'calculate_envelope_activity' function: case 6
def test_calculate_envelope_activity_invalid_plot_results():
    # Test with invalid plot_results value
    input_signal = np.array([1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="The plotting results must be 0 or 1."):
        calculate_envelope_activity(input_signal, plot_results=2)


# Test function for the 'find_consecutive_groups' function: case 1
def test_find_consecutive_groups():
    """
    Test for find_consecutive_groups function in the 'ngmt.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = np.array([0, 1, 1, 0, 2, 2, 2, 0, 0, 3, 3])

    # Call the find_consecutive_groups function with the specified inputs
    ind = find_consecutive_groups(test_signal)

    # Assertions to be checked:
    # Check that the input signal is a NumPy array
    assert isinstance(test_signal, np.ndarray), "Input signal should be a NumPy array."

    # Check that the input signal is not empty
    assert test_signal.size >= 1, "Input signal must not be empty."

    # Check that the output is a NumPy array
    assert isinstance(ind, np.ndarray), "Output should be a NumPy array."

    # Check that the output has the correct shape
    assert ind.shape[1] == 2, "Output should have 2 columns."

    if ind.size > 0:
        # Check that the start index is less than or equal to the end index for each row
        assert np.all(
            ind[:, 0] <= ind[:, 1]
        ), "Start index should be less than or equal to end index."

        # Check that the output contains only non-negative values
        assert np.all(ind >= 0), "Output indices contain negative values."

        # Check that the indices point to valid positions in the input signal
        assert np.all(ind[:, 0] < test_signal.size), "Start index is out of bounds."
        assert np.all(ind[:, 1] < test_signal.size), "End index is out of bounds."

        # Check that the output indices represent consecutive non-zero groups
        for start, end in ind:
            assert np.all(
                test_signal[start : end + 1] != 0
            ), "Non-zero values not consecutive."


# Test function for the 'find_consecutive_groups' function: case 2
def test_find_consecutive_groups_valid_input():
    # Test with valid input
    input_signal = np.array([0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7, 8])

    ind = find_consecutive_groups(input_signal)

    # Check that the output is a NumPy array
    assert isinstance(ind, np.ndarray)

    # Check that the shape of the output is correct
    assert ind.shape[1] == 2

    # Check that the output does not contain any NaN or Inf values
    assert not np.isnan(ind).any(), "The output contains NaN values."
    assert not np.isinf(ind).any(), "The output contains Inf values."

    # Check the correctness of the output
    expected_result = np.array([[1, 2], [5, 7], [11, 11], [13, 14]])
    np.testing.assert_array_equal(ind, expected_result)


# Test function for the 'find_consecutive_groups' function: case 3
def test_find_consecutive_groups_empty_input():
    # Test with empty input data
    input_signal = np.array([])

    with pytest.raises(ValueError, match="Input data must not be empty."):
        find_consecutive_groups(input_signal)


# Test function for the 'find_consecutive_groups' function: case 4
def test_find_consecutive_groups_invalid_input():
    # Test with invalid input data type
    input_signal = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="Input data must be a NumPy array."):
        find_consecutive_groups(input_signal)


# Test function for the 'find_consecutive_groups' function: case 5
def test_find_consecutive_groups_single_value():
    # Test with a single non-zero value
    input_signal = np.array([0, 0, 0, 5, 0, 0, 0])

    ind = find_consecutive_groups(input_signal)

    # Check that the output is a NumPy array
    assert isinstance(ind, np.ndarray)

    # Check that the shape of the output is correct
    assert ind.shape[1] == 2

    # Check that the output does not contain any NaN or Inf values
    assert not np.isnan(ind).any(), "The output contains NaN values."
    assert not np.isinf(ind).any(), "The output contains Inf values."

    # Check the correctness of the output
    expected_result = np.array([[3, 3]])
    np.testing.assert_array_equal(ind, expected_result)


# Test function for the 'find_consecutive_groups' function: case 6
def test_find_consecutive_groups_large_input():
    # Test with a large input array
    input_signal = np.zeros(1000)
    input_signal[100:200] = 1
    input_signal[300:400] = 2
    input_signal[700:800] = 3

    ind = find_consecutive_groups(input_signal)

    # Check that the output is a NumPy array
    assert isinstance(ind, np.ndarray)

    # Check that the shape of the output is correct
    assert ind.shape[1] == 2

    # Check that the output does not contain any NaN or Inf values
    assert not np.isnan(ind).any(), "The output contains NaN values."
    assert not np.isinf(ind).any(), "The output contains Inf values."

    # Check the correctness of the output
    expected_result = np.array([[100, 199], [300, 399], [700, 799]])
    np.testing.assert_array_equal(ind, expected_result)


# Test function for the 'find_local_min_max' function: case 1
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


# Test function for the 'find_local_min_max' function: case 2
def test_find_local_min_max_empty_input():
    # Test with empty input data
    signal = np.array([])

    with pytest.raises(ValueError, match="Input signal must not be empty."):
        find_local_min_max(signal)


# Test function for the 'find_local_min_max' function: case 2
def test_find_local_min_max_invalid_input():
    # Test with invalid input data type
    signal = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="Input signal must be a NumPy array."):
        find_local_min_max(signal)


# Test function for the 'find_local_min_max' function: case 3
def test_find_local_min_max_no_minima():
    # Test with no minima
    signal = np.array([1, 3, 5, 7, 9])

    minima_indices, maxima_indices = find_local_min_max(signal)

    # Check that the output is an empty array for minima
    assert minima_indices.size == 0

    # Check that the output is a NumPy array for maxima
    assert isinstance(maxima_indices, np.ndarray)


# Test function for the 'find_local_min_max' function: case 4
def test_find_local_min_max_no_maxima():
    # Test with no maxima
    signal = np.array([9, 7, 5, 3, 1])

    minima_indices, maxima_indices = find_local_min_max(signal)

    # Check that the output is an empty array for maxima
    assert maxima_indices.size == 0

    # Check that the output is a NumPy array for minima
    assert isinstance(minima_indices, np.ndarray)


# Test function for the 'identify_pulse_trains' function: case 3
def test_identify_pulse_trains_single_element():
    # Test with a single element in the input data
    signal = np.array([42])

    pulse_trains = identify_pulse_trains(signal)

    # Check that the output is an empty list
    assert pulse_trains == []


# Test function for the 'identify_pulse_trains' function: case 4
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


# Test function for the 'identify_pulse_trains' function: case 5
def test_identify_pulse_trains_empty_signal():
    # Create an empty input signal
    signal = np.array([])

    # Call the function with the empty signal and expect a ValueError
    with pytest.raises(ValueError) as exc_info:
        identify_pulse_trains(signal)

    # Check if the correct error message is raised
    assert str(exc_info.value) == "Input signal must not be empty."


# Test function for the 'convert_pulse_train_to_array' function: case 1
def test_convert_pulse_train_to_array():
    """
    Test for convert_pulse_train_to_array function in the 'ngmt.utils.preprocessing' module.
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

    # Test for an empty input list
    with pytest.raises(ValueError, match="Input list is empty."):
        convert_pulse_train_to_array([])

    # Test for a list with a dictionary missing 'start' key
    invalid_list_missing_start = [{"end": 10}, {"start": 20, "end": 30}]
    with pytest.raises(
        ValueError, match="Each dictionary should contain 'start' and 'end' keys."
    ):
        convert_pulse_train_to_array(invalid_list_missing_start)

    # Test for a list with a dictionary missing 'end' key
    invalid_list_missing_end = [{"start": 0}, {"start": 20, "end": 30}]
    with pytest.raises(
        ValueError, match="Each dictionary should contain 'start' and 'end' keys."
    ):
        convert_pulse_train_to_array(invalid_list_missing_end)


# Test function for the 'convert_pulse_train_to_array' function: case 2
def test_convert_pulse_train_to_array_valid_input():
    # Test with valid input
    pulse_train_list = [
        {"start": 1, "end": 3},
        {"start": 7, "end": 9},
        {"start": 15, "end": 17},
        {"start": 21, "end": 23},
    ]

    array_representation = convert_pulse_train_to_array(pulse_train_list)

    # Check that the output is a NumPy array
    assert isinstance(array_representation, np.ndarray)

    # Check the shape of the array
    assert array_representation.shape == (len(pulse_train_list), 2)

    # Check the correctness of the array
    expected_array = np.array([[1, 3], [7, 9], [15, 17], [21, 23]], dtype=np.uint64)
    np.testing.assert_array_equal(array_representation, expected_array)


# Test function for the 'convert_pulse_train_to_array' function: case 3
def test_convert_pulse_train_to_array_empty_input():
    # Test with empty input list
    pulse_train_list = []

    with pytest.raises(ValueError, match="Input list is empty."):
        convert_pulse_train_to_array(pulse_train_list)


# Test function for the 'convert_pulse_train_to_array' function: case 4
def test_convert_pulse_train_to_array_invalid_input_type():
    # Test with invalid input type
    pulse_train_list = "not_a_list"

    with pytest.raises(
        ValueError, match="Input should be a list of pulse train dictionaries."
    ):
        convert_pulse_train_to_array(pulse_train_list)


# Test function for the 'convert_pulse_train_to_array' function: case 5
def test_convert_pulse_train_to_array_invalid_element_type():
    # Test with invalid element type in the list
    pulse_train_list = [{"start": 1, "end": 3}, "not_a_dictionary"]

    with pytest.raises(
        ValueError, match="Each element in the list should be a dictionary."
    ):
        convert_pulse_train_to_array(pulse_train_list)


# Test function for the 'convert_pulse_train_to_array' function: case 6
def test_convert_pulse_train_to_array_missing_keys():
    # Test with dictionaries missing 'start' or 'end' keys
    pulse_train_list = [{"start": 1, "end": 3}, {"start": 7}]

    with pytest.raises(
        ValueError, match="Each dictionary should contain 'start' and 'end' keys."
    ):
        convert_pulse_train_to_array(pulse_train_list)


# Test function for the 'convert_pulse_train_to_array' function: case 7
def test_convert_pulse_train_to_array_invalid_key_names():
    # Test with dictionaries having invalid key names
    pulse_train_list = [{"begin": 1, "finish": 3}, {"start": 7, "end": 9}]

    with pytest.raises(
        ValueError, match="Each dictionary should contain 'start' and 'end' keys."
    ):
        convert_pulse_train_to_array(pulse_train_list)


# Test function for the 'find_interval_intersection' function: case 1
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

    # Additional Test Case 1: Empty sets
    set_a_empty = np.array([])
    set_b_empty = np.array([])
    expected_result_empty = np.array([])

    # Assertions to be checked:
    # Check if set_a_empty and set_b_empty are NumPy arrays
    assert isinstance(set_a_empty, np.ndarray), "set_a_empty should be a NumPy array."
    assert isinstance(set_b_empty, np.ndarray), "set_b_empty should be a NumPy array."

    try:
        # Call the find_interval_intersection function with the specified inputs
        result_empty = find_interval_intersection(set_a_empty, set_b_empty)

        # Check the data type of the output
        assert isinstance(result_empty, np.ndarray), "Output should be a NumPy array."

        # Check if the output matches the expected result
        npt.assert_array_equal(
            result_empty,
            expected_result_empty,
            "Output does not match the expected result.",
        )

    except IndexError:
        # Handle the case where an IndexError occurs (empty array)
        assert (
            len(set_a_empty) == 0 and len(set_b_empty) == 0
        ), "Empty arrays should result in empty output."

    # Additional Test Case 2: Identical sets
    set_a_identical = np.array([[1, 5], [7, 10]])
    set_b_identical = np.array([[1, 5], [7, 10]])
    expected_result_identical = np.array([[1, 5], [7, 10]])

    # Assertions to be checked:
    # Check if set_a_identical and set_b_identical are NumPy arrays
    assert isinstance(
        set_a_identical, np.ndarray
    ), "set_a_identical should be a NumPy array."
    assert isinstance(
        set_b_identical, np.ndarray
    ), "set_b_identical should be a NumPy array."

    # Check if set_a_identical and set_b_identical have the correct structure (two columns)
    assert set_a_identical.shape[1] == 2, "set_a_identical should have two columns."
    assert set_b_identical.shape[1] == 2, "set_b_identical should have two columns."

    # Call the find_interval_intersection function with the specified inputs
    result_identical = find_interval_intersection(set_a_identical, set_b_identical)

    # Check the data type of the output
    assert isinstance(result_identical, np.ndarray), "Output should be a NumPy array."


# Test function for the 'find_interval_intersection' function: case 2
def test_find_interval_intersection_valid_input():
    # Test with valid input
    set_a = np.array([[1, 4], [6, 9], [11, 15]])
    set_b = np.array([[2, 5], [8, 12]])

    intersection_intervals = find_interval_intersection(set_a, set_b)

    # Check that the output is a NumPy array
    assert isinstance(intersection_intervals, np.ndarray)

    # Check the shape of the array
    assert intersection_intervals.shape[1] == 2

    # Check the correctness of the array
    expected_result = np.array([[2, 4], [8, 9], [11, 12]])
    np.testing.assert_array_equal(intersection_intervals, expected_result)


# Test function for the 'find_interval_intersection' function: case 3
def test_find_interval_intersection_invalid_input_type():
    # Test with invalid input types
    set_a = "not_a_numpy_array"
    set_b = np.array([[1, 4], [6, 9]])

    with pytest.raises(ValueError, match="Both input sets should be NumPy arrays."):
        find_interval_intersection(set_a, set_b)


# Test function for the 'find_interval_intersection' function: case 4
def test_find_interval_intersection_invalid_set_structure():
    # Test with invalid set structure (more than two columns)
    set_a = np.array([[1, 4, 7], [6, 9, 12]])
    set_b = np.array([[2, 5], [8, 12]])

    with pytest.raises(
        ValueError,
        match="Input sets should have two columns, indicating start and end points.",
    ):
        find_interval_intersection(set_a, set_b)


# Test function for the 'find_interval_intersection' function: case 5
def test_find_interval_intersection_append_from_set_b():
    # Test case where an interval from set B is appended to the intersection intervals
    set_a = np.array([[1, 5]])
    set_b = np.array([[3, 6], [7, 9]])

    # Call the function
    result = find_interval_intersection(set_a, set_b)

    # Expecting the interval [3, 5] from set B to be appended
    expected_result = np.array([[3, 5]])
    assert np.array_equal(result, expected_result)


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


# Test function for the 'organize_and_pack_results' function
def test_step_time_calculation_no_peak_steps():
    # Mock input data
    walking_periods = [(0, 10)]
    peak_steps = []

    # Call the function
    organized_results, _ = organize_and_pack_results(walking_periods, peak_steps)

    # Expecting the step time calculation to not affect the result
    assert organized_results[0]["start"] == 0
    assert organized_results[0]["end"] == 10


# Test function for the 'max_peaks_between_zc' function
def test_max_peaks_between_zc_valid_input():
    # Test with a valid non-empty input signal
    input_signal = np.array([1, -2, 3, -4, 5, -6, 7, -8, 9])
    pks, ipks = max_peaks_between_zc(input_signal)

    # Check that the outputs are NumPy arrays
    assert isinstance(pks, np.ndarray)
    assert isinstance(ipks, np.ndarray)

    # Check that the lengths of pks and ipks are consistent
    assert len(pks) == len(ipks)

    # Check that ipks are within the valid range of indices for the input signal
    assert np.all((ipks >= 0) & (ipks < len(input_signal)))

    # Retrieve the signed max/min values at the peak locations.
    pks_retrieved = input_signal[ipks.astype(int) - 1]
    assert np.all(pks == pks_retrieved)


# Test function for the 'signal_decomposition_algorithm' function: case 1
def test_signal_decomposition_algorithm_invalid_input_type():
    # Test with invalid input type
    vertical_acceleration_data = "not_a_numpy_array"
    initial_sampling_frequency = 100

    # Check that the function raises an error for invalid input type
    with pytest.raises(
        ValueError, match="vertical_acceleration_data must be a numpy.ndarray"
    ):
        signal_decomposition_algorithm(
            vertical_acceleration_data, initial_sampling_frequency
        )


# Test function for the 'signal_decomposition_algorithm' function: case 2
def test_signal_decomposition_algorithm_negative_sampling_frequency():
    # Test with negative initial sampling frequency
    vertical_acceleration_data = np.array([1, 2, 3, 4, 5])
    initial_sampling_frequency = -100

    # Check that the function raises an error for negative sampling frequency
    with pytest.raises(
        ValueError, match="The initial sampling frequency must be a positive float."
    ):
        signal_decomposition_algorithm(
            vertical_acceleration_data, initial_sampling_frequency
        )


# Test function for the 'signal_decomposition_algorithm' function: case 3
def test_invalid_input_data():
    # Test case for invalid input data type
    with pytest.raises(ValueError):
        signal_decomposition_algorithm("invalid")


# Test function for the 'signal_decomposition_algorithm' function: case 3
def test_at_least_one_dimension():
    # Test case for input data with less than one dimension
    with pytest.raises(ValueError):
        signal_decomposition_algorithm(np.array(1))


# Test function for the 'classify_physical_activity' function: case 2
def test_classify_physical_activity_invalid_input_data():
    # Test with invalid input data type
    input_data = "not_a_dataframe"
    sedentary_threshold = 0.2
    light_threshold = 0.5
    moderate_threshold = 0.8
    epoch_duration = 5

    with pytest.raises(ValueError, match="Input_data must be a pandas DataFrame."):
        classify_physical_activity(
            input_data,
            sedentary_threshold,
            light_threshold,
            moderate_threshold,
            epoch_duration,
        )


# Test function for the 'classify_physical_activity' function: case 3
def test_classify_physical_activity_invalid_threshold_type():
    # Test with invalid threshold type
    input_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    sedentary_threshold = "invalid_type"
    light_threshold = 0.5
    moderate_threshold = 0.8
    epoch_duration = 5

    with pytest.raises(ValueError, match="Threshold values must be numeric."):
        classify_physical_activity(
            input_data,
            sedentary_threshold,
            light_threshold,
            moderate_threshold,
            epoch_duration,
        )


# Test function for the 'classify_physical_activity' function: case 5
def test_classify_physical_activity_invalid_threshold_values():
    invalid_data = pd.DataFrame(
        {
            "timestamps": pd.date_range(start="2024-01-01", periods=100, freq="S"),
            "enmo": np.random.rand(100) * 500,
        }
    )

    # Call the classify_physical_activity function with invalid threshold values
    with pytest.raises(ValueError, match="Threshold values must be numeric."):
        classify_physical_activity(invalid_data, sedentary_threshold="invalid")


# Test function for the 'classify_physical_activity' function: case 6
def test_classify_physical_activity_negative_epoch_duration():
    invalid_data = pd.DataFrame(
        {
            "timestamps": pd.date_range(start="2024-01-01", periods=100, freq="S"),
            "enmo": np.random.rand(100) * 500,
        }
    )

    # Call the classify_physical_activity function with negative epoch_duration
    with pytest.raises(ValueError, match="Epoch_duration must be a positive integer."):
        classify_physical_activity(invalid_data, epoch_duration=-5)


@pytest.fixture
def sample_accelerometer_data():
    # Create sample accelerometer data
    np.random.seed(0)
    timestamps = pd.date_range(start="2024-01-01", periods=1000, freq="1s")
    accelerometer_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": np.random.randn(1000),
            "LowerBack_ACCEL_y": np.random.randn(1000),
            "LowerBack_ACCEL_z": np.random.randn(1000),
        },
        index=timestamps,
    )
    return accelerometer_data


@pytest.fixture
def sample_gait_sequences():
    # Create sample gait sequences DataFrame
    gait_sequences = pd.DataFrame(
        {"onset": [1.5, 3.5, 5.5], "duration": [0.5, 0.7, 0.6]}
    )
    return gait_sequences


def test_detect_method(sample_accelerometer_data, sample_gait_sequences):
    # Initialize ParaschivIonescuInitialContactDetection instance
    icd_instance = ParaschivIonescuInitialContactDetection()

    # Call detect method
    icd_instance.detect(
        data=sample_accelerometer_data,
        gait_sequences=sample_gait_sequences,
        sampling_freq_Hz=100,
    )

    # Check if initial_contacts_ attribute is a DataFrame
    assert isinstance(icd_instance.initial_contacts_, pd.DataFrame)

    # Check the columns in the initial_contacts_ DataFrame
    expected_columns = ["onset", "event_type", "tracking_systems", "tracked_points"]
    assert all(
        col in icd_instance.initial_contacts_.columns for col in expected_columns
    )

    # Check the data type of the 'onset' column
    assert pd.api.types.is_float_dtype(icd_instance.initial_contacts_["onset"])

    # Check if onset values are within the expected range
    assert all(0 <= onset <= 6 for onset in icd_instance.initial_contacts_["onset"])

## Module test
# Test data
num_samples = 50000  # Number of samples
acceleration_data = {
    "LowerBack_ACCEL_x": np.random.uniform(-2, 2, num_samples),
    "LowerBack_ACCEL_y": np.random.uniform(-2, 2, num_samples),
    "LowerBack_ACCEL_z": np.random.uniform(-2, 2, num_samples),
}
acceleration_data = pd.DataFrame(acceleration_data)
sampling_frequency = 100  # Sampling frequency


def test_gsd_detect():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Call the detect method
    gsd.detect(data=acceleration_data, sampling_freq_Hz=sampling_frequency)
    gait_sequences_ = gsd.gait_sequences_

    # Assertions
    assert isinstance(
        gait_sequences_, pd.DataFrame
    ), "Gait sequences should be a DataFrame."
    assert (
        "onset" in gait_sequences_.columns
    ), "Gait sequences should have 'onset' column."
    assert (
        "duration" in gait_sequences_.columns
    ), "Gait sequences should have 'duration' column."
    assert (
        "event_type" in gait_sequences_.columns
    ), "Gait sequences should have 'event_type' column."
    assert (
        "tracking_systems" in gait_sequences_.columns
    ), "Gait sequences should have 'tracking_systems' column."
    assert (
        "tracked_points" in gait_sequences_.columns
    ), "Gait sequences should have 'tracked_points' column."


def test_invalid_input_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid input data
    invalid_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    with pytest.raises(ValueError):
        gsd.detect(data=invalid_data, sampling_freq_Hz=sampling_frequency)


def test_invalid_sampling_freq():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid sampling frequency
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(data=acceleration_data, sampling_freq_Hz=invalid_sampling_freq)


def test_gait_sequence_detection():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Call the detect method
    gsd.detect(data=acceleration_data, sampling_freq_Hz=sampling_frequency)

    # Check if gait_sequences_ attribute is a DataFrame
    assert isinstance(
        gsd.gait_sequences_, pd.DataFrame
    ), "Gait sequences should be a DataFrame."

    # Check if gait_sequences_ DataFrame has the expected columns
    expected_columns = [
        "onset",
        "duration",
        "event_type",
        "tracking_systems",
        "tracked_points",
    ]
    assert all(
        col in gsd.gait_sequences_.columns for col in expected_columns
    ), "Gait sequences DataFrame should have the expected columns."

    # Check if all onset values are within the correct range
    assert all(
        onset >= 0 and onset <= acceleration_data.shape[0] / sampling_frequency
        for onset in gsd.gait_sequences_["onset"]
    ), "Onset values should be within the valid range."

    # Check if all duration values are non-negative
    assert all(
        duration >= 0 for duration in gsd.gait_sequences_["duration"]
    ), "Duration values should be non-negative."


def test_invalid_input_data_type():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid input data type
    invalid_data = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        gsd.detect(data=invalid_data, sampling_freq_Hz=sampling_frequency)


def test_invalid_sampling_freq_type():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid sampling frequency type
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(data=acceleration_data, sampling_freq_Hz=invalid_sampling_freq)


def test_plot_results_type():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid plot_results type
    invalid_plot_results = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            plot_results=invalid_plot_results,
        )


# Tests for ParaschivIonescuInitialContactDetection
def test_detect_empty_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect with an empty DataFrame instead of None
    icd.detect(data=pd.DataFrame(), gait_sequences=pd.DataFrame(), sampling_freq_Hz=100)


# Define test_detect_no_gait_sequences function
def test_detect_no_gait_sequences():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    icd = ParaschivIonescuInitialContactDetection()

    # Create a DataFrame with only one column for each axis
    acceleration_data_single_axis = {
        "LowerBack_ACCEL_x": np.random.uniform(-2, 2, num_samples),
        "LowerBack_ACCEL_y": np.random.uniform(-2, 2, num_samples),
        "LowerBack_ACCEL_z": np.random.uniform(-2, 2, num_samples),
    }
    acceleration_data_single_axis = pd.DataFrame(acceleration_data_single_axis)

    # Call detect without gait sequences
    icd.detect(
        data=acceleration_data_single_axis,
        gait_sequences=pd.DataFrame(),
        sampling_freq_Hz=100,
    )


def test_detect_no_plot():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    icd = ParaschivIonescuInitialContactDetection()

    # Create a DataFrame with only one column for each axis
    acceleration_data_single_axis = {
        "LowerBack_ACCEL_x": np.random.uniform(-2, 2, num_samples),
        "LowerBack_ACCEL_y": np.random.uniform(-2, 2, num_samples),
        "LowerBack_ACCEL_z": np.random.uniform(-2, 2, num_samples),
    }
    acceleration_data_single_axis = pd.DataFrame(acceleration_data_single_axis)

    # Call detect without gait sequences
    icd.detect(
        data=acceleration_data_single_axis,
        gait_sequences=pd.DataFrame(),
        sampling_freq_Hz=100,
    )

    # Check if initial_contacts_ is None
    assert (
        icd.initial_contacts_ is None
    ), "Initial contacts should be None if no gait sequences are provided"


# Test data
num_samples = 50000  # Number of samples
acceleration_data = {
    "LARM_ACCEL_x": np.random.uniform(-2, 2, num_samples),
    "LARM_ACCEL_y": np.random.uniform(-2, 2, num_samples),
    "LARM_ACCEL_z": np.random.uniform(-2, 2, num_samples),
}
acceleration_data = pd.DataFrame(acceleration_data)
sampling_frequency = 100  # Sampling frequency
time_index = pd.date_range(
    start="2024-02-07", periods=num_samples, freq=f"{1/sampling_frequency}S"
)
acceleration_data["timestamp"] = time_index
acceleration_data.set_index("timestamp", inplace=True)


# Tests for PhysicalActivityMonitoring
def test_pam_detect():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Call the detect method
    pam.detect(
        data=acceleration_data,
        sampling_freq_Hz=sampling_frequency,
        thresholds_mg={
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec=5,
        plot_results=False,  # Set to False to avoid plotting for this test
    )
    physical_activities_ = pam.physical_activities_

    # Assertions
    assert isinstance(
        physical_activities_, pd.DataFrame
    ), "Physical activity information should be stored in a DataFrame."


def test_invalid_sampling_freq_pam():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid sampling frequency
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=invalid_sampling_freq,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
        )


def test_invalid_thresholds_type():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid thresholds type
    invalid_thresholds = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg=invalid_thresholds,
            epoch_duration_sec=5,
        )


def test_invalid_epoch_duration():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid epoch duration
    invalid_epoch_duration = -1
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=invalid_epoch_duration,
        )


def test_invalid_plot_results_type_pam():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid plot_results type
    invalid_plot_results = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot_results=invalid_plot_results,
        )


def test_invalid_sampling_freq_type_error_handling():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid sampling frequency type
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=invalid_sampling_freq,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot_results=True,
        )


def test_invalid_thresholds_type_error_handling():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid thresholds type
    invalid_thresholds = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg=invalid_thresholds,
            epoch_duration_sec=5,
            plot_results=True,
        )


def test_empty_input_data():
    # Define empty_data with required columns
    empty_data = pd.DataFrame(
        {
            "LARM_ACCEL_x": [],
            "LARM_ACCEL_y": [],
            "LARM_ACCEL_z": [],
        }
    )

    # Initialize the PhysicalActivityMonitoring class
    pam = PhysicalActivityMonitoring()

    # Call the detect method with empty_data
    with pytest.raises(ValueError):
        pam.detect(data=empty_data, sampling_freq_Hz=sampling_frequency)


def test_single_data_point():
    pam = PhysicalActivityMonitoring()
    single_data_point = pd.DataFrame(
        {"LARM_ACCEL_x": [0], "LARM_ACCEL_y": [1], "LARM_ACCEL_z": [2]},
        index=[pd.Timestamp("2024-02-07 00:00:00")],
    )
    with pytest.raises(ValueError):
        pam.detect(data=single_data_point, sampling_freq_Hz=sampling_frequency)


def test_pam_detect_full_coverage():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Call the detect method with plot_results=False to avoid plotting
    pam.detect(
        data=acceleration_data,
        sampling_freq_Hz=sampling_frequency,
        thresholds_mg={
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec=5,
        plot_results=False,
    )
    physical_activities_ = pam.physical_activities_

    # Assertions
    assert isinstance(
        physical_activities_, pd.DataFrame
    ), "Physical activity information should be stored in a DataFrame."

    # Check if the DataFrame has expected columns
    expected_columns = [
        "date",
        "sedentary_mean_enmo",
        "sedentary_time_min",
        "light_mean_enmo",
        "light_time_min",
        "moderate_mean_enmo",
        "moderate_time_min",
        "vigorous_mean_enmo",
        "vigorous_time_min",
    ]
    assert all(
        col in physical_activities_.columns for col in expected_columns
    ), "DataFrame should have the expected columns."


def test_plot_results():

    # Initialize PhysicalActivityMonitoring instance
    pam = PhysicalActivityMonitoring()

    # Define sample parameters
    sampling_freq_Hz = 100
    thresholds_mg = {
        "sedentary_threshold": 45,
        "light_threshold": 100,
        "moderate_threshold": 400,
    }
    epoch_duration_sec = 5
    plot_results = True  # Set to True to test plotting

    # Call detect method
    result = pam.detect(
        data=acceleration_data,
        sampling_freq_Hz=sampling_frequency,
        thresholds_mg=thresholds_mg,
        epoch_duration_sec=epoch_duration_sec,
        plot_results=plot_results,
    )

    # Save the figure as a temporary file
    temp_file = "test_plot_results.png"
    plt.savefig(temp_file)

    # Close the figure
    plt.close()

    # Cleanup the temporary file
    import os

    os.remove(temp_file)


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
