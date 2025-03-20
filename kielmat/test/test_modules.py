# Introduction and explanation regarding the test suite
"""
This code is a test suite for various signal processing and analysis functions which exist in the KielMAT toolbox.
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
import os
from kielmat.modules.gsd import ParaschivIonescuGaitSequenceDetection
from kielmat.modules.icd import ParaschivIonescuInitialContactDetection
from kielmat.modules.pam import PhysicalActivityMonitoring
from kielmat.modules.ptd import PhamPosturalTransitionDetection
from kielmat.modules.td import PhamTurnDetection

## Module test
# Test for gait sequence detection algorithm
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
    gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=sampling_frequency)
    gait_sequences_ = gsd.gait_sequences_


def test_invalid_sampling_freq():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid sampling frequency
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=invalid_sampling_freq)


def test_gait_sequence_detection():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Call the detect method
    gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=sampling_frequency)


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
        gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=invalid_sampling_freq)


def test_plot_results_type():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid plot_results type
    invalid_plot_results = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(
            accel_data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            plot_results=invalid_plot_results,
        )


# Test for ValueError: "dt_data must be a pandas Series with datetime values"
def test_invalid_dt_data_type():
    gsd = ParaschivIonescuGaitSequenceDetection()
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_y": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_z": np.random.uniform(-2, 2, 1000),
        }
    )
    with pytest.raises(ValueError):
        gsd.detect(
            accel_data=acceleration_data, sampling_freq_Hz=100, dt_data="not_a_series"
        )


# Test for ValueError: "dt_data must be a series with the same length as data"
def test_invalid_dt_data_length():
    gsd = ParaschivIonescuGaitSequenceDetection()
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_y": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_z": np.random.uniform(-2, 2, 1000),
        }
    )
    dt_data = pd.Series(
        pd.date_range(start="2022-01-01", periods=500)
    )  # Different length than data
    with pytest.raises(ValueError):
        gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=100, dt_data=dt_data)


def test_threshold_selected_signal():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Create empty data
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": [],
            "LowerBack_ACCEL_y": [],
            "LowerBack_ACCEL_z": [],
        }
    )
    dt_data = pd.Series([])

    # Call detect with empty data
    with pytest.raises(ValueError):
        gsd.detect(
            accel_data=acceleration_data,
            sampling_freq_Hz=100,
            dt_data=dt_data,
            plot_results=True,
        )


def test_no_gait_sequences_detected():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Create empty data
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": [],
            "LowerBack_ACCEL_y": [],
            "LowerBack_ACCEL_z": [],
        }
    )

    # Call detect with empty data
    with pytest.raises(ValueError):
        gsd.detect(accel_data=acceleration_data, sampling_freq_Hz=100)


def test_invalid_indices_warning():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Create random acceleration data and datetime series with more indices than data length
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_y": np.random.uniform(-2, 2, 1000),
            "LowerBack_ACCEL_z": np.random.uniform(-2, 2, 1000),
        }
    )
    dt_data = pd.Series(
        pd.date_range(start="2022-01-01", periods=1000)
    )  # Same length as data


def test_no_plotting_datetime_values():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Create empty data
    acceleration_data = pd.DataFrame(
        {
            "LowerBack_ACCEL_x": [],
            "LowerBack_ACCEL_y": [],
            "LowerBack_ACCEL_z": [],
        }
    )

    # Create an empty pandas Series (no datetime values)
    dt_data = pd.Series([])

    # Call detect with empty data and datetime series without datetime values
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        gsd.detect(
            accel_data=acceleration_data,
            sampling_freq_Hz=100,
            dt_data=dt_data,
            plot_results=True,
        )


# Tests for initial contact detection algorithm
def test_detect_empty_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect with an empty DataFrame instead of None
    icd.detect(
        accel_data=pd.DataFrame(),
        gait_sequences=pd.DataFrame(),
        sampling_freq_Hz=100,
        v_acc_col_name="LowerBack",
    )


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
        accel_data=acceleration_data_single_axis,
        gait_sequences=pd.DataFrame(),
        sampling_freq_Hz=100,
        v_acc_col_name="LowerBack_ACCEL_y",
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
        accel_data=acceleration_data_single_axis,
        gait_sequences=pd.DataFrame(),
        sampling_freq_Hz=100,
        v_acc_col_name="LowerBack_ACCEL_y",
    )

    # Check if initial_contacts_ is None
    assert (
        icd.initial_contacts_ is None
    ), "Initial contacts should be None if no gait sequences are provided"


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
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect method
    icd.detect(
        accel_data=sample_accelerometer_data,
        gait_sequences=sample_gait_sequences,
        sampling_freq_Hz=100,
        v_acc_col_name="LowerBack_ACCEL_y",
    )

    # Check if initial_contacts_ attribute is a DataFrame
    assert isinstance(icd.initial_contacts_, pd.DataFrame)

    # Check the columns in the initial_contacts_ DataFrame
    expected_columns = ["onset", "event_type", "tracking_systems"]
    assert all(col in icd.initial_contacts_.columns for col in expected_columns)

    # Check the data type of the 'onset' column
    assert pd.api.types.is_float_dtype(icd.initial_contacts_["onset"])

    # Check if onset values are within the expected range
    assert all(0 <= onset <= 6 for onset in icd.initial_contacts_["onset"])


def test_detect_method_invalid_dt_data_type(
    sample_accelerometer_data, sample_gait_sequences
):
    # Initialize ParaschivIonescuInitialContactDetection instance
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect method with invalid dt_data type
    with pytest.raises(ValueError) as excinfo:
        icd.detect(
            accel_data=sample_accelerometer_data,
            gait_sequences=sample_gait_sequences,
            sampling_freq_Hz=100,
            v_acc_col_name="LowerBack_ACCEL_y",
            dt_data="not a series",
        )
    assert str(excinfo.value) == "dt_data must be a pandas Series with datetime values"


def test_detect_method_invalid_dt_data_length(
    sample_accelerometer_data, sample_gait_sequences
):
    # Initialize ParaschivIonescuInitialContactDetection instance
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect method with dt_data not having the same length as data
    with pytest.raises(ValueError) as excinfo:
        icd.detect(
            accel_data=sample_accelerometer_data,
            gait_sequences=sample_gait_sequences,
            sampling_freq_Hz=100,
            v_acc_col_name="LowerBack_ACCEL_y",
            dt_data=pd.Series([1, 2, 3, 4]),
        )
    assert str(excinfo.value) == "dt_data must be a pandas Series with datetime values"


# Tests for phyisical activity monitoring algorithm
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
    start="2024-02-07", periods=num_samples, freq=f"{1/sampling_frequency}s"
)
acceleration_data["timestamp"] = time_index
acceleration_data.set_index("timestamp", inplace=True)


def test_pam_detect():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Call the detect method
    pam.detect(
        data=acceleration_data,
        acceleration_unit="m/s^2",
        sampling_freq_Hz=sampling_frequency,
        thresholds_mg={
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec=5,
        plot=False,  # Set to False to avoid plotting for this test
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
            acceleration_unit="m/s^2",
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
            acceleration_unit="m/s^2",
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
            acceleration_unit="m/s^2",
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
            acceleration_unit="m/s^2",
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot=invalid_plot_results,
        )


def test_invalid_sampling_freq_type_error_handling():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid sampling frequency type
    invalid_sampling_freq = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            acceleration_unit="m/s^2",
            sampling_freq_Hz=invalid_sampling_freq,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot=True,
        )


def test_invalid_thresholds_type_error_handling():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid thresholds type
    invalid_thresholds = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=acceleration_data,
            acceleration_unit="m/s^2",
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg=invalid_thresholds,
            epoch_duration_sec=5,
            plot=True,
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
        pam.detect(
            data=empty_data,
            acceleration_unit="m/s^2",
            sampling_freq_Hz=sampling_frequency,
        )


def test_pam_detect_full_coverage():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Call the detect method with plot_results=False to avoid plotting
    pam.detect(
        data=acceleration_data,
        acceleration_unit="m/s^2",
        sampling_freq_Hz=sampling_frequency,
        thresholds_mg={
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec=5,
        plot=False,
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


# Test function for test_PhysicalActivityMonitoring
def test_PhysicalActivityMonitoring():
    """
    Test for PhysicalActivityMonitoring class.
    """
    # Generate some sample accelerometer data
    num_samples = 50000  # Number of samples
    acceleration_data = {
        "LARM_ACCEL_x": np.random.uniform(-2, 2, num_samples),
        "LARM_ACCEL_y": np.random.uniform(-2, 2, num_samples),
        "LARM_ACCEL_z": np.random.uniform(-2, 2, num_samples),
    }
    acceleration_data = pd.DataFrame(acceleration_data)
    acceleration_unit = "m/s^2"
    sampling_frequency = 100  # Sampling frequency
    time_index = pd.date_range(
        start="2024-02-07", periods=num_samples, freq=f"{1/sampling_frequency}s"
    )
    acceleration_data["timestamp"] = time_index
    acceleration_data.set_index("timestamp", inplace=True)

    # Initialize PhysicalActivityMonitoring instance
    pam = PhysicalActivityMonitoring()

    # Test detect method
    pam.detect(
        data=acceleration_data,
        acceleration_unit="m/s^2",
        sampling_freq_Hz=100,
        thresholds_mg={
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec=5,
        plot=False,
    )

    # Assertions for physical_activities_ attribute
    assert isinstance(
        pam.physical_activities_, pd.DataFrame
    ), "physical_activities_ should be a DataFrame."


def test_invalid_input_data_type():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with invalid input data type
    invalid_data = "invalid"
    with pytest.raises(ValueError):
        pam.detect(
            data=invalid_data,
            acceleration_unit="m/s^2",
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot=False,
        )


def test_insufficient_columns():
    # Initialize the class
    pam = PhysicalActivityMonitoring()

    # Test with insufficient columns
    data_with_insufficient_columns = acceleration_data[["LARM_ACCEL_x", "LARM_ACCEL_y"]]
    with pytest.raises(ValueError):
        pam.detect(
            data=data_with_insufficient_columns,
            acceleration_unit="m/s^2",
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg={
                "sedentary_threshold": 45,
                "light_threshold": 100,
                "moderate_threshold": 400,
            },
            epoch_duration_sec=5,
            plot=False,
        )


# Test functions for Turn detection algorithm
def test_pham_turn_detection_algorithm():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)
    sampling_freq = 200

    # Check that the target sampling frequency is positive
    assert (
        sampling_freq > 0
    ), "Assertion Error: Sampling frequency should be greater than 0."

    # Check that the input data is a pandas DataFrame
    assert isinstance(
        sample_data, pd.DataFrame
    ), "Input data must be a pandas DataFrame"

    # Perform detection
    pham.detect(
        accel_data=sample_data.iloc[:, 0:3],
        gyro_data=sample_data.iloc[:, 3:6],
        gyro_vertical="pelvis_GYRO_x",
        sampling_freq_Hz=sampling_freq,
        plot_results=False,
    )

    # Perform spatio-temporal detection
    pham.spatio_temporal_parameters()


def test_invalid_plot_results_pham_td():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # Test with invalid plot_results
    invalid_plot_results = "invalid"
    with pytest.raises(ValueError):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            gyro_vertical="pelvis_GYRO_x",
            sampling_freq_Hz=200,
            plot_results=invalid_plot_results,
        )


def test_invalid_sampling_freq_pham():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # Test with non-positive sampling frequency
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            gyro_vertical="pelvis_GYRO_x",
            sampling_freq_Hz=0,
        )


def test_invalid_dt_data():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)
    dt_data = pd.Series(
        pd.date_range("20210101", periods=500)
    )  # Invalid length for datetime data

    # Test with invalid datetime data
    with pytest.raises(
        ValueError, match="dt_data must be a series with the same length as data"
    ):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            gyro_vertical="pelvis_GYRO_x",
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )


def test_dt_data_invalid_type():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # dt_data is not a pandas Series
    dt_data = np.array(pd.date_range(start="1/1/2022", periods=1000, freq="s"))

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            gyro_vertical="pelvis_GYRO_x",
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )


def test_dt_data_invalid_dtype():
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # dt_data is a pandas Series but not with datetime dtype
    dt_data = pd.Series(np.arange(1000))

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            gyro_vertical="pelvis_GYRO_x",
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )


# Test functions for pham postural transition algorithm
def test_pham_postural_transition_algorithm():
    # Initialize PhamPosturalTransitionDetection object
    pham = PhamPosturalTransitionDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    input_data = pd.read_csv(test_data_file_path, index_col=0)
    sampling_freq = 200

    # Perform detection with valid input data
    pham.detect(
        accel_data=input_data.iloc[:, 0:3],
        gyro_data=input_data.iloc[:, 3:6],
        sampling_freq_Hz=sampling_freq,
        plot_results=False,
    )

    # Perform spatio-temporal parameter extraction
    pham.spatio_temporal_parameters()

    # Test 1: dt_data is a pandas Series but not with datetime dtype
    dt_data = pd.Series(np.arange(1000))

    # Should raise ValueError because dt_data is not datetime
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        pham.detect(
            accel_data=input_data.iloc[:, 0:3],
            gyro_data=input_data.iloc[:, 3:6],
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )

    # Test 2: dt_data is a NumPy array instead of a pandas Series
    dt_data = np.array(pd.date_range(start="1/1/2022", periods=1000, freq="s"))

    # Should raise ValueError because dt_data is not a pandas Series
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        pham.detect(
            accel_data=input_data.iloc[:, 0:3],
            gyro_data=input_data.iloc[:, 3:6],
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )

    # Test 3: dt_data is a pandas Series of datetime values but of incorrect length
    dt_data = pd.Series(pd.date_range("20210101", periods=500))

    # Should raise ValueError because dt_data length does not match input_data length
    with pytest.raises(
        ValueError,
        match="dt_data must be a pandas Series with the same length as accel_data and gyro_data",
    ):
        pham.detect(
            accel_data=input_data.iloc[:, 0:3],
            gyro_data=input_data.iloc[:, 3:6],
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )

    # Test 4: dt_data is a pandas Series but converted to a NumPy array
    dt_data = pd.Series(pd.date_range(start="1/1/2022", periods=1000, freq="s"))
    dt_data = dt_data.to_numpy()

    # Should raise ValueError because dt_data is not a pandas Series
    with pytest.raises(
        ValueError, match="dt_data must be a pandas Series with datetime values"
    ):
        pham.detect(
            accel_data=input_data.iloc[:, 0:3],
            gyro_data=input_data.iloc[:, 3:6],
            sampling_freq_Hz=200,
            dt_data=dt_data,
        )


def test_data_structure_invalid_pham_pt():
    # Initialize PhamPosturalTransitionDetection object
    pham = PhamPosturalTransitionDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # Extract accelerometer and gyroscope data correctly
    accel_data = sample_data.iloc[:, 0:3]
    gyro_data = sample_data.iloc[:, 3:6]

    # Convert DataFrames to numpy arrays
    accel_data = accel_data.to_numpy()
    gyro_data = gyro_data.to_numpy()

    # Test invalid data shape
    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame"):
        pham.detect(
            accel_data=accel_data,
            gyro_data=gyro_data,
            sampling_freq_Hz=100,
        )


def test_invalid_sampling_freq_pham_pt():
    # Initialize PhamPosturalTransitionDetection object
    pham = PhamPosturalTransitionDetection()

    # Sample accelerometer and gyroscope data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_file_path = os.path.join(current_dir, "test_data_pham_algorithms.csv")
    sample_data = pd.read_csv(test_data_file_path, index_col=0)

    # Test with non-positive sampling frequency
    with pytest.raises(ValueError, match="Sampling frequency must be positive"):
        pham.detect(
            accel_data=sample_data.iloc[:, 0:3],
            gyro_data=sample_data.iloc[:, 3:6],
            sampling_freq_Hz=0,
        )


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
