# Introduction and explanation regarding the test suite
"""
This code is a test suite for various signal processing and analysis functions which exist in the KielMAT toolbox.
It employs pytest, a Python testing framework, to verify the correctness of these functions.
Here's a brief explanation of the code structure:

1. Import necessary libraries, pytest and the functions to be tested.
2. Generate a random input signal for testing purposes.
3. Define a series of test functions, each targeting a specific function from the 'kielmat.utils.preprocessing' module.
4. Inside each test function, we validate the correctness of the corresponding function and its inputs.
5. We make use of 'assert' statements to check that the functions return expected results.
6. In some test functions for plot functions, the "monkeypatch" is used which is a feature provided by the pytest library.
It allows to temporarily modify or replace attributes, methods, or functions during testing. Here, monkeypatch is being
used to replace the plt.show() function. This way, when run test functions where it won't actually display any plots,
allowing to focus on testing the logic of the function without the overhead of rendering plots.
7. The code is organized for clarity and maintainability.

To run the tests, follow these steps:

1. Make sure you have pytest installed. If not, install it using 'pip install -U pytest'.
2. Run this script, and pytest will execute all the test functions.
3. Any failures in tests will be reported as failed with red color, and also the number of passed tests will be represented with green color.

By running these tests, the reliability and correctness of the signal processing functions in the 'kielmat.utils.preprocessing' module will be ensured.
"""

# Import necessary libraries and functions to be tested.
import pandas as pd
import numpy as np
import matplotlib as plt

plt.use("Agg")
import warnings
import numpy.testing as npt
import pytest
from pathlib import Path
from kielmat.utils.kielmat_dataclass import KielMATRecording
from bids_validator import BIDSValidator
from kielmat.utils import viz_utils
from kielmat.utils.preprocessing import (
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
    tilt_angle_estimation,
    wavelet_decomposition,
    moving_var,
)
from kielmat.utils.quaternion import (
    quatinv,
    quatnormalize,
    quatnorm,
    quatconj,
    quatmultiply,
    rotm2quat,
    quat2rotm,
    quat2axang,
    axang2rotm,
)

# Generate a random sinusoidal signal with varying amplitudes to use as an input in testing functions
time = np.linspace(0, 100, 1000)  # Time vector from 0 to 100 with 1000 samples
amplitudes = np.random.uniform(-3, 3, 1000)  # Random amplitudes between 3 and -3
sampling_frequency = 100  # Sampling frequency
random_input_signal = np.sin(2 * np.pi * sampling_frequency * time) * amplitudes


# Each test function checks a specific function.
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
    Test for lowpass_filter_savgol function in the 'kielmat.utils.preprocessing' module.
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
    Test for lowpass_filter_fir function in the 'kielmat.utils.preprocessing' module.
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
    Test specific parameters for Savitzky-Golay filter in the 'kielmat.utils.preprocessing' module.
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
    Test specific parameters for Butterworth filter in the 'kielmat.utils.preprocessing' module.
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
    """Test for highpass_filter_iir function in the 'kielmat.utils.preprocessing' module."""
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
    """Test for _iir_highpass_filter function in the 'kielmat.utils.preprocessing' module."""
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
    Test for apply_continuous_wavelet_transform function in the 'kielmat.utils.preprocessing' module.
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
    Test for apply_successive_gaussian_filters function in the 'kielmat.utils.preprocessing' module.
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
    Test for calculate_envelope_activity function in the 'kielmat.utils.preprocessing' module.
    """
    # Test with inputs
    test_signal = random_input_signal
    smooth_window = 20
    threshold_style = 1
    duration = 20

    # Call the calculate_envelope_activity function with the specified inputs
    alarm, env = calculate_envelope_activity(
        test_signal, smooth_window, threshold_style, duration
    )


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


# Test function for the 'find_consecutive_groups' function: case 1
def test_find_consecutive_groups():
    """
    Test for find_consecutive_groups function in the 'kielmat.utils.preprocessing' module.
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
    Test for find_local_min_max function in the 'kielmat.utils.preprocessing' module.
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
    Test for convert_pulse_train_to_array function in the 'kielmat.utils.preprocessing' module.
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
    Test for convert_pulse_train_to_array function in the 'kielmat.utils.preprocessing' module.
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
    Test for organize_and_pack_results function in the 'kielmat.utils.preprocessing' module.
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
    time_column_name = "timestamp"
    sedentary_threshold = 0.2
    light_threshold = 0.5
    moderate_threshold = 0.8
    epoch_duration = 5

    with pytest.raises(ValueError, match="Input_data must be a pandas DataFrame."):
        classify_physical_activity(
            input_data,
            time_column_name,
            sedentary_threshold,
            light_threshold,
            moderate_threshold,
            epoch_duration,
        )


# Test function for the 'classify_physical_activity' function: case 3
def test_classify_physical_activity_invalid_threshold_type():
    # Test with invalid threshold type
    input_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
    time_column_name = "timestamp"
    sedentary_threshold = "invalid_type"
    light_threshold = 0.5
    moderate_threshold = 0.8
    epoch_duration = 5

    with pytest.raises(ValueError, match="Threshold values must be numeric."):
        classify_physical_activity(
            input_data,
            time_column_name,
            sedentary_threshold,
            light_threshold,
            moderate_threshold,
            epoch_duration,
        )


# Test function for the 'classify_physical_activity' function: case 5
def test_classify_physical_activity_invalid_threshold_values():
    invalid_data = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="s"),
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
            "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="s"),
            "enmo": np.random.rand(100) * 500,
        }
    )

    # Call the classify_physical_activity function with negative epoch_duration
    with pytest.raises(ValueError, match="Epoch_duration must be a positive integer."):
        classify_physical_activity(invalid_data, epoch_duration=-5)


# Test function for wavelet_decomposition function
def test_wavelet_decomposition():
    """
    Test for wavelet_decomposition function in the 'kielmat.utils.preprocessing' module.
    """
    # Generate a random input signal
    input_signal = np.random.randn(1000)

    # Test with valid inputs
    denoised_signal = wavelet_decomposition(input_signal, level=3, wavetype="haar")

    # Assertions
    assert isinstance(
        denoised_signal, np.ndarray
    ), "Denoised signal should be a NumPy array."
    assert len(denoised_signal) == len(
        input_signal
    ), "Denoised signal length should match input signal length."
    assert not np.isnan(denoised_signal).any(), "Denoised signal contains NaN values."
    assert not np.isinf(
        denoised_signal
    ).any(), "Denoised signal contains infinite values."


# Test function for moving_var function
def test_moving_var():
    """
    Test for moving_var function in the 'kielmat.utils.preprocessing' module.
    """
    # Generate a random input signal
    input_signal = np.random.randn(1000)

    # Test with valid inputs
    moving_variance = moving_var(input_signal, window=10)

    # Assertions
    assert isinstance(
        moving_variance, np.ndarray
    ), "Moving variance should be a NumPy array."
    assert len(moving_variance) == len(
        input_signal
    ), "Moving variance length should match input signal length."
    assert not np.isnan(moving_variance).any(), "Moving variance contains NaN values."
    assert not np.isinf(
        moving_variance
    ).any(), "Moving variance contains infinite values."


# Test function for test_tilt_angle_estimation function
def test_tilt_angle_estimation():
    """
    Test for tilt_angle_estimation function.
    """
    # Generate some sample gyro data
    gyro_data = np.array(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]]
    )

    sampling_frequency_hz = 10  # Sampling frequency of 10 Hz

    # Calculate expected tilt angle
    expected_tilt_angle = np.array([-0.02, -0.05, -0.1, -0.18])  # Manually calculated

    # Test tilt_angle_estimation function
    tilt_angle = tilt_angle_estimation(gyro_data, sampling_frequency_hz)

    # Assertions
    assert isinstance(tilt_angle, np.ndarray), "Tilt angle should be a NumPy array."

    # Test with DataFrame input
    gyro_df = pd.DataFrame(gyro_data)
    tilt_angle_df = tilt_angle_estimation(gyro_df, sampling_frequency_hz)

    # Assertions for DataFrame input
    assert isinstance(
        tilt_angle_df, np.ndarray
    ), "Tilt angle from DataFrame should be a NumPy array."

    # Test for invalid input type
    with pytest.raises(
        TypeError, match="Input data must be a numpy array or pandas DataFrame"
    ):
        tilt_angle_estimation(
            list(gyro_data), sampling_frequency_hz
        )  # Passing a list instead of numpy array


# Test gsd_plot_results without plotting
# Sample data for testing
target_sampling_freq_Hz = 100
detected_activity_signal = np.random.rand(1000)
gait_sequences_ = pd.DataFrame(
    {"onset": np.array([100, 300, 500]), "duration": np.array([50, 60, 70])}
)
hourly_average_data = pd.DataFrame(
    np.random.rand(24, 7),
    columns=pd.date_range(start="2024-01-01", periods=7),
    index=np.arange(24),
)
thresholds_mg = {
    "sedentary_threshold": 45,
    "light_threshold": 100,
    "moderate_threshold": 400,
}


# Test function for gsd_plot_results without plotting uisng monkeypatch which allows testing function without plotting
def test_gsd_plot_results_without_plot(monkeypatch):
    # A mock function for plt.show() that does nothing
    def mock_show():
        pass

    # Monkeypatch plt.show() with the mock function
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    # Call the function
    viz_utils.plot_gait(
        target_sampling_freq_Hz, detected_activity_signal, gait_sequences_
    )


# Test function for pam_plot_results without plotting
def test_pam_plot_results_without_plot(monkeypatch):
    # A mock function for plt.show() that does nothing
    def mock_show():
        pass

    # Monkeypatch plt.show() with the mock function
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    # Call the function
    viz_utils.plot_pam(hourly_average_data)


# Test function for test_pham_plot_results
def test_pham_plot_results(monkeypatch):
    # Generate sample data
    np.random.seed(0)
    accel = np.random.randn(100, 3)
    gyro = np.random.randn(100, 3)

    # Ensure 'onset' and 'duration' arrays have the same length
    onset = np.arange(10, 90, 20)
    duration = [5] * len(onset)

    postural_transitions_ = pd.DataFrame({"onset": onset, "duration": duration})

    sampling_freq_Hz = 100

    # A mock function for plt.show() that does nothing
    def mock_show():
        pass

    # Monkeypatch plt.show() with the mock function
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    # Call the function
    viz_utils.plot_postural_transitions(
        accel,
        gyro,
        postural_transitions_=postural_transitions_,
        sampling_freq_Hz=sampling_freq_Hz,
    )


# Test function for test_organize_and_pack_results
@pytest.mark.parametrize(
    "walking_periods, peak_steps, expected_results",
    [
        # Test case 1
        (
            [(0, 20)],  # Walking periods
            [2, 6, 10, 12, 15, 20],  # Peak steps
            [
                {
                    "start": -2,
                    "end": 22,
                    "steps": 6,
                    "mid_swing": [2, 6, 10, 12, 15, 20],
                }
            ],  # Expected results
        ),
        # Test case 2
        (
            [(0, 15), (16, 30)],  # Walking periods
            [0, 2, 6, 10, 12, 15, 20, 25, 26, 28, 30],  # Peak steps
            [
                {
                    "start": -1,
                    "end": 31,
                    "steps": 11,
                    "mid_swing": [0, 2, 6, 10, 12, 15, 20, 25, 26, 28, 30],
                }
            ],  # Expected results
        ),
    ],
)
# Test function for test_organize_and_pack_results
def test_organize_and_pack_results(walking_periods, peak_steps, expected_results):
    # Call the function and get the actual results
    actual_results, actual_peak_steps = organize_and_pack_results(
        walking_periods, peak_steps
    )

    # Sort the peak steps in the actual results
    actual_peak_steps.sort()

    # Flatten the nested lists
    actual_results_flat = [
        item for sublist in actual_results for item in sublist.items()
    ]


# Generate test data
quaternions = np.array(
    [
        [0.5, 0.5, 0.5, 0.5],  # Quaternion 1
        [1.0, 0.0, 0.0, 0.0],  # Quaternion 2
        [0.707, 0.0, 0.707, 0.0],  # Quaternion 3
    ]
)
rotation_matrices = np.array(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # Rotation Matrix 1
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],  # Rotation Matrix 2
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],  # Rotation Matrix 3
    ]
)
axis_angle_rep = np.array(
    [
        [1, 0, 0, np.pi / 2],  # Axis-Angle Representation 1
        [1, 0, 0, 0],  # Axis-Angle Representation 2
        [0, 1, 0, np.pi / 2],  # Axis-Angle Representation 3
    ]
)


# Test function for quatinv function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[0], np.array([0.5, -0.5, -0.5, -0.5])),  # Test case 1
        (quaternions[1], np.array([1.0, 0.0, 0.0, 0.0])),  # Test case 2
        (quaternions[2], np.array([0.707, 0.0, -0.707, 0.0])),  # Test case 3
    ],
)
def test_quatinv(q, expected):
    result = quatinv(q)


# Test function for quatnormalize function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[0], np.array([0.5, 0.5, 0.5, 0.5])),  # Test case 1
        (quaternions[1], np.array([1.0, 0.0, 0.0, 0.0])),  # Test case 2
        (quaternions[2], np.array([0.707, 0.0, 0.707, 0.0])),  # Test case 3
    ],
)
def test_quatnormalize(q, expected):
    result = quatnormalize(q)


# Test function for quatnorm function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[0], np.array([1.0] * 3)),  # Test case 1
        (quaternions[1], np.array([1.0] * 3)),  # Test case 2
        (quaternions[2], np.array([1.0] * 3)),  # Test case 3
    ],
)
def test_quatnorm(q, expected):
    result = quatnorm(q)


# Test function for quatconj function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[0], np.array([0.5, -0.5, -0.5, -0.5])),  # Test case 1
        (quaternions[1], np.array([1.0, 0.0, 0.0, 0.0])),  # Test case 2
        (quaternions[2], np.array([0.707, 0.0, -0.707, 0.0])),  # Test case 3
    ],
)
def test_quatconj(q, expected):
    result = quatconj(q)


# Test function for quatmultiply function
@pytest.mark.parametrize(
    "q1, q2, expected",
    [
        (quaternions[0], quaternions[1], np.array([0.0, 1.0, 0.0, 0.0])),  # Test case 1
        (
            quaternions[1],
            quaternions[2],
            np.array([0.707, 0.0, 0.0, -0.707]),
        ),  # Test case 2
        (
            quaternions[2],
            quaternions[0],
            np.array([0.0, 0.0, -0.707, 0.707]),
        ),  # Test case 3
    ],
)
def test_quatmultiply(q1, q2):
    result = quatmultiply(q1, q2)


# Test function for rotm2quat function
@pytest.mark.parametrize(
    "R, expected",
    [
        (rotation_matrices[0], quaternions[1]),  # Test case 1
        (rotation_matrices[1], quaternions[2]),  # Test case 2
        (rotation_matrices[2], np.array([0.924, 0.383, 0.0, 0.0])),  # Test case 3
    ],
)
def test_rotm2quat(R, expected):
    result = rotm2quat(R)


# Test function for quat2rotm function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[1], rotation_matrices[0]),  # Test case 1
        (quaternions[2], rotation_matrices[1]),  # Test case 2
        (np.array([0.924, 0.383, 0.0, 0.0]), rotation_matrices[2]),  # Test case 3
    ],
)
def test_quat2rotm(q, expected):
    result = quat2rotm(q)


# Test function for quat2axang function
@pytest.mark.parametrize(
    "q, expected",
    [
        (quaternions[1], axis_angle_rep[1]),  # Test case 1
        (quaternions[0], axis_angle_rep[0]),  # Test case 2
        (quaternions[2], axis_angle_rep[2]),  # Test case 3
    ],
)
def test_quat2axang(q, expected):
    result = quat2axang(q)


# Test function for axang2rotm function
@pytest.mark.parametrize(
    "axang, expected",
    [
        (axis_angle_rep[0], rotation_matrices[0]),  # Test case 1
        (axis_angle_rep[1], rotation_matrices[1]),  # Test case 2
        (axis_angle_rep[2], rotation_matrices[2]),  # Test case 3
    ],
)
def test_axang2rotm(axang, expected):
    result = axang2rotm(axang)


# Test function for quatconj function with different configurations
@pytest.mark.parametrize(
    "q, scalar_first, channels_last, expected",
    [
        (
            np.array([[1, 0, 0, 0]]),
            True,
            True,
            np.array([[1, 0, 0, 0]]),
        ),  # Identity quaternion
        (
            np.array([[0, 1, 0, 0]]),
            True,
            True,
            np.array([[0, -1, 0, 0]]),
        ),  # Pure imaginary quaternion
        (
            np.array([[0, 0, 1, 0]]),
            True,
            False,
            np.array([[0, 0, -1, 0]]),
        ),  # Pure imaginary quaternion, channels_last = False
        (
            np.array([[0, 0, 0, 1]]),
            False,
            True,
            np.array([[0, 0, 0, -1]]),
        ),  # Pure imaginary quaternion, scalar_last = True
        (
            np.array([[[0, 1, 0, 0], [0, 0, 1, 0]]]),
            True,
            True,
            np.array([[[0, -1, 0, 0], [0, 0, -1, 0]]]),
        ),  # Two quaternions
    ],
)
def case_two_test_quatconj(q, scalar_first, channels_last, expected):
    result = quatconj(q, scalar_first=scalar_first, channels_last=channels_last)


# Test function for quatconj function
def test_quatconj_transpose():
    """Test quatconj with transposed quaternion."""
    q = np.array([[[0, 1, 0, 0], [0, 0, 1, 0]]])
    result = quatconj(q, scalar_first=True, channels_last=False)


def test_quatconj_manipulation():
    """Test quatconj with quaternion manipulation."""
    q = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    q_tmp = q.copy()
    q[..., 0] = q_tmp[..., -1]
    q[..., 1:] = q_tmp[..., :-1]
    del q_tmp
    result = quatconj(q, scalar_first=False, channels_last=True)


# Test function for quatmultiply function with different configurations
@pytest.mark.parametrize(
    "q1, q2",
    [
        (
            np.array([[[1, 0, 0, 0]]]),
            np.array([[[1, 0, 0, 0]]]),
        ),  # Identity quaternion
        (
            np.array([[[0, 1, 0, 0]]]),
            np.array([[[0, 0, 1, 0]]]),
        ),  # Pure imaginary quaternions
    ],
)
def test_quatmultiply(q1, q2):
    result = quatmultiply(q1, q2)


# Test function for quatmultiply function with channels_last=True
@pytest.mark.parametrize(
    "q1, q2",
    [
        (
            np.array(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
            ),  # q1 with channels_last=True
            np.array(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
            ),  # q2 with channels_last=True
        ),
        (
            np.array(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
            ),  # q1 with channels_last=True
            np.array(
                [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]]
            ),  # q2 with channels_last=True
        ),
    ],
)
def test_quatmultiply_channels_last(q1, q2):

    # Determine channels_last based on the shape of q1
    channels_last = q1.shape[-1]

    result = quatmultiply(
        q1, q2, channels_last=channels_last
    )  # Pass channels_last accordingly


# Test function for rotm2quat function with different methods
def test_method_copysign():
    """Test rotm2quat with method 'copysign'."""
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    expected = np.array([0.70710678, 0.70710678, 0, 0])
    result = rotm2quat(R, method="copysign")


def test_invalid_method():
    """Test rotm2quat with an invalid method."""
    R = np.eye(3)
    with pytest.raises(
        RuntimeError, match='invalid method, must be "copysign", "auto", 0, 1, 2 or 3'
    ):
        rotm2quat(R, method=4)


# Test cases for quatmultiply function
@pytest.mark.parametrize(
    "q1_shape, q2_shape, scalar_first, channels_last",
    [
        ((3, 4), (3, 4), True, True),  # Basic case
        ((3, 4), (3, 4), False, True),  # Test scalar_last=True
        ((3, 4), None, True, True),  # Test self-multiplication
        ((3, 1, 4), (3, 1, 4), True, True),  # Test broadcasting
    ],
)
def test_quatmultiply(q1_shape, q2_shape, scalar_first, channels_last):
    # Generate random quaternion arrays with given shapes
    q1 = np.random.rand(*q1_shape)
    q2 = None if q2_shape is None else np.random.rand(*q2_shape)

    # Call the quatmultiply function
    result = quatmultiply(
        q1, q2, scalar_first=scalar_first, channels_last=channels_last
    )


# Test function for axang2rotm function
@pytest.mark.parametrize(
    "axang, expected",
    [
        (np.array([0.15, 0.25, 0.35, 0.0]), np.eye(3)),  # Test case with correct shape
    ],
)
def test_axang2rotm(axang, expected):
    # Reshape axang to have the required shape (..., 4)
    axang = axang.reshape(-1, 4)
    result = axang2rotm(axang)
    assert np.allclose(result, expected)


# Test function for without plotting
def test_pham_turn_plot_results_no_plot(monkeypatch):
    # a mock function for plt.show() that does nothing
    def mock_show():
        pass

    # Generate mock data
    accel = np.random.rand(100, 3)  # Acceleration data
    gyro = np.random.rand(100, 3)  # Gyroscope data
    detected_turns = pd.DataFrame(
        {"onset": [10.5, 20.0, 35.2], "duration": [2.0, 1.5, 3.0]}
    )  # Detected turns DataFrame
    sampling_freq_Hz = 50  # Sampling frequency

    # Monkeypatch plt.show() with the mock function
    monkeypatch.setattr("matplotlib.pyplot.show", mock_show)

    # Call the function
    viz_utils.plot_turns(accel, gyro, detected_turns, sampling_freq_Hz)


# Fixture to provide sample data for testing
@pytest.fixture
def sample_data():
    """Fixture to provide a sample DataFrame with accelerometer data."""
    return pd.DataFrame(
        {
            "Time": [0.1, 0.2, 0.3],
            "ACCEL_x": [1.0, 2.0, 3.0],
            "ACCEL_y": [0.5, 1.5, 2.5],
            "ACCEL_z": [0.2, 0.3, 0.4],
        }
    )


# Fixture to provide sample channel information for testing
@pytest.fixture
def sample_channels():
    """Fixture to provide a sample DataFrame with channel information."""
    return pd.DataFrame(
        {
            "name": ["ACCEL_x", "ACCEL_y", "ACCEL_z"],  # Channel names
            "component": ["x", "y", "z"],  # Component of the measurement
            "type": ["ACCEL", "ACCEL", "ACCEL"],  # Type of measurement
            "tracked_point": ["n/a", "n/a", "n/a"],  # Point being tracked
            "units": ["m/s^2", "m/s^2", "m/s^2"],  # Units of measurement
            "sampling_frequency": [100, 100, 100],  # Sampling frequency
        }
    )


# Fixture to provide sample event data for testing
@pytest.fixture
def sample_events():
    """Fixture to provide a sample DataFrame with event information."""
    return pd.DataFrame(
        {
            "onset": [0.1, 0.2],  # Onset times of events
            "duration": [0.1, 0.2],  # Durations of events
            "event_type": ["stimulus", "response"],  # Type of events
            "name": ["Stimulus A", "Response B"],  # Names of events
        }
    )


# Fixture to create a sample KielMATRecording object for testing
@pytest.fixture
def sample_recording(sample_data, sample_channels):
    """Fixture to create a sample KielMATRecording object with sample data and channels."""
    return KielMATRecording(
        data={
            "tracking_system_1": sample_data
        },  # Sample data under one tracking system
        channels={
            "tracking_system_1": sample_channels
        },  # Sample channel info under one tracking system
    )


# Test to validate channel information in the recording
def test_validate_channels_valid(sample_recording):
    """Test the validation of channel dataframes in the recording."""
    try:
        message = sample_recording.validate_channels()
        assert message == "All channel dataframes are valid."
    except ValueError:
        pytest.fail("Validation failed for valid channels")


# Test to add events to the recording
def test_add_events(sample_recording, sample_events):
    """Test adding events to the recording and verify their presence."""
    recording = sample_recording
    recording.add_events(tracking_system="tracking_system_1", new_events=sample_events)
    assert (
        "tracking_system_1" in recording.events
    )  # Check if the tracking system has events
    assert len(recording.events["tracking_system_1"]) == len(
        sample_events
    )  # Check if the number of events matches


# Test to add general info to the recording
def test_add_info(sample_recording):
    """Test adding general info to the recording."""
    recording = sample_recording
    recording.add_info("Subject", "001")


# Test to handle adding info with an invalid key
def test_add_info_invalid_key(sample_recording):
    """Test adding info with an invalid key."""
    recording = sample_recording
    recording.add_info("InvalidKey", "001")


# Test to handle adding info with case conversion
def test_add_info_case_conversion(sample_recording):
    """Test adding info with case conversion in the value."""
    recording = sample_recording
    recording.add_info("Subject", "SubJECT01")


# Test to handle adding info with underscores removed
def test_add_info_remove_underscores(sample_recording):
    """Test adding info with underscores removed from the value."""
    recording = sample_recording
    recording.add_info("Task", "Task_Name")


# Test to export events to a CSV file without BIDS compatibility
def test_export_events_single_system(sample_recording, tmp_path):
    """Test exporting events to a CSV file for a single tracking system without BIDS compatibility."""
    recording = sample_recording
    file_path = tmp_path / "tracking_system_1_events.csv"
    recording.export_events(
        file_path=str(file_path),
        tracking_system="tracking_system_1",
        bids_compatible_fname=False,
    )


# Test to export events to a CSV file with BIDS compatibility
def test_export_events_bids_compatible(sample_recording, tmp_path):
    """Test exporting events to a CSV file with BIDS-compatible filename."""
    recording = sample_recording
    recording.add_info("Subject", "001")
    recording.add_info("Task", "task01")
    file_path = tmp_path / "sub-001_task-task01_events.csv"
    recording.export_events(file_path=str(file_path), bids_compatible_fname=True)


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
