# Introduction and explanation regarding the test suite
"""
This code is a test suite for various signal processing and analysis functions which exist in the NGMT toolbox. 
It employs pytest, a Python testing framework, to verify the correctness of these functions. 
Here's a brief explanation of the code structure:

1. Import necessary libraries, pytest and the functions to be tested.
2. Generate a random input signal for testing purposes.
3. Define a series of test functions, each targeting a specific function from the specific module.
4. Inside each test module, we validate the correctness of the corresponding function and its inputs.
5. We make use of 'assert' statements to check that the functions return expected results.
6. The code is organized for clarity and maintainability.

To run the tests, follow these steps:

1. Make sure you have pytest installed. If not, install it using 'pip install -U pytest'.
2. Run this script, and pytest will execute all the test functions.
3. Any failures in tests will be reported as failed with red color, and also the number of passed tests will be represented with green color.

By running these tests, the reliability and correctness of the modules will be ensured.
"""

# Import necessary libraries and modules to be tested.
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection
from ngmt.modules.pam import PhysicalActivityMonitoring
from matplotlib.testing.compare import compare_images

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



# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
