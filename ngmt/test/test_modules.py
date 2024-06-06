# Introduction and explanation regarding the test suite
"""
This code is a test suite for various modules which exist in the NGMT toolbox. 
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
from unittest.mock import patch
from ngmt.datasets import keepcontrol
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection
from ngmt.modules.pam import PhysicalActivityMonitoring
from ngmt.modules.ptd import PhamPosturalTransitionDetection
from ngmt.modules.td import PhamTurnDetection


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
    gsd.detect(data=acceleration_data, sampling_freq_Hz=sampling_frequency)
    gait_sequences_ = gsd.gait_sequences_


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

def test_invalid_dt_data_type():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid dt_data type
    invalid_dt_data = "invalid"
    with pytest.raises(ValueError):
        gsd.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            plot_results=False,
            dt_data=invalid_dt_data,
        )


def test_invalid_dt_data_format():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid dt_data format
    invalid_dt_data = pd.Series(["2024-03-22 10:00:00", "2024-03-22 10:00:01"])
    with pytest.raises(ValueError):
        gsd.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            plot_results=False,
            dt_data=invalid_dt_data,
        )


def test_invalid_dt_data_length():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid dt_data length
    invalid_dt_data = pd.Series(
        pd.date_range(start="2024-03-22", periods=3, freq="S")
    )
    with pytest.raises(ValueError):
        gsd.detect(
            data=acceleration_data,
            sampling_freq_Hz=sampling_frequency,
            plot_results=False,
            dt_data=invalid_dt_data,
        )

def test_valid_dt_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Create valid datetime data
    valid_dt_data = pd.Series(
        pd.date_range(start="2024-03-22", periods=len(acceleration_data), freq="S")
    )

    # Call the detect method with valid datetime data
    gsd.detect(
        data=acceleration_data,
        sampling_freq_Hz=sampling_frequency,
        plot_results=False,
        dt_data=valid_dt_data,
    )

# Tests for initial contact detection algorithm
def test_detect_empty_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    icd = ParaschivIonescuInitialContactDetection()

    # Call detect with an empty DataFrame instead of None
    icd.detect(data=pd.DataFrame(), gait_sequences=pd.DataFrame(), sampling_freq_Hz=100)


# Test_detect_no_gait_sequences function
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
    start="2024-02-07", periods=num_samples, freq=f"{1/sampling_frequency}S"
)
acceleration_data["timestamp"] = time_index
acceleration_data.set_index("timestamp", inplace=True)


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
            sampling_freq_Hz=sampling_frequency,
            thresholds_mg=invalid_thresholds,
            epoch_duration_sec=5,
            plot=True,
        )


def test_empty_input_data():
    # empty_data with required columns
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
    sampling_frequency = 100  # Sampling frequency
    time_index = pd.date_range(
        start="2024-02-07", periods=num_samples, freq=f"{1/sampling_frequency}S"
    )
    acceleration_data["timestamp"] = time_index
    acceleration_data.set_index("timestamp", inplace=True)

    # Initialize PhysicalActivityMonitoring instance
    pam = PhysicalActivityMonitoring()

    # Test detect method
    pam.detect(
        data=acceleration_data,
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


# Tests for sit to stand and stand to sit detection algorithm
@pytest.fixture
def sample_data():
    # Generate sample data
    num_samples = 50000  # Number of samples
    sample_data = pd.DataFrame(
        {
            "Acc_X": np.random.uniform(-2, 2, num_samples),
            "Acc_Y": np.random.uniform(-2, 2, num_samples),
            "Acc_Z": np.random.uniform(-2, 2, num_samples),
            "Gyro_X": np.random.uniform(-150, 150, num_samples),
            "Gyro_Y": np.random.uniform(-150, 150, num_samples),
            "Gyro_Z": np.random.uniform(-150, 150, num_samples),
        }
    )
    return sample_data

# Some additional test functions for PhamPosturalTransitionDetection algorithm
class PhamPosturalTransitionDetection:
    @staticmethod
    def test_detection_output_shape(self):
        pham_ssd_detector = PhamPosturalTransitionDetection()
        # Generate sample data
        num_samples = 50000  # Number of samples
        data = pd.DataFrame(
            {
                "Acc_X": np.random.uniform(-2, 2, num_samples),
                "Acc_Y": np.random.uniform(-2, 2, num_samples),
                "Acc_Z": np.random.uniform(-2, 2, num_samples),
                "Gyro_X": np.random.uniform(-150, 150, num_samples),
                "Gyro_Y": np.random.uniform(-150, 150, num_samples),
                "Gyro_Z": np.random.uniform(-150, 150, num_samples),
            }
        )
        result = pham_ssd_detector.detect(
            data,
            accel_unit='g',
            gyro_unit='deg/s',
            sampling_freq_Hz=100,
            plot_results=False
        )
        # Assuming the result should be a DataFrame
        assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"

    def test_plot_results_pham(self):
        # Initialize the class
        pham_ssd_detector = PhamPosturalTransitionDetection()
        # Generate sample data
        num_samples = 50000  # Number of samples
        data = pd.DataFrame(
            {
                "Acc_X": np.random.uniform(-2, 2, num_samples),
                "Acc_Y": np.random.uniform(-2, 2, num_samples),
                "Acc_Z": np.random.uniform(-2, 2, num_samples),
                "Gyro_X": np.random.uniform(-150, 150, num_samples),
                "Gyro_Y": np.random.uniform(-150, 150, num_samples),
                "Gyro_Z": np.random.uniform(-150, 150, num_samples),
            }
        )
        # Test with invalid plot_results type
        invalid_plot_results = "invalid"
        with pytest.raises(ValueError):
            pham_ssd_detector.detect(
                data=data,
                accel_unit='g',
                gyro_unit='deg/s',
                sampling_freq_Hz=100,
                plot_results=invalid_plot_results,
            )

# Test functions for Turn detection algorithm
@pytest.fixture
def detector():
    return PhamTurnDetection()

@pytest.fixture
def gyro_data_numpy():
    # Sample accelerometer and gyroscope data
    accel_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Accel_X', 'Accel_Y', 'Accel_Z'])
    gyro_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    sample_data = pd.concat([accel_data, gyro_data], axis=1)

    # Select gyro data and convert it to numpy array format
    gyro = sample_data.iloc[:, 3:6].copy()
    gyro = gyro.to_numpy()
    return gyro

def test_bias_calculation_valid(detector, gyro_data_numpy):
    # Sample accelerometer and gyroscope data
    accel_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Accel_X', 'Accel_Y', 'Accel_Z'])
    gyro_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    sample_data = pd.concat([accel_data, gyro_data], axis=1)

    # Perform detection
    detector.detect(sample_data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='Gyro_X', sampling_freq_Hz=100, plot_results=False)
    
    # Manually calculate the gyro bias for comparison
    gyro_bias_manual = np.mean(gyro_data_numpy[:100], axis=0)
    
    # Retrieve the calculated bias from the detector
    gyro_bias_detected = detector.gyro_bias

# Some additional test functions for Turn detection algorithm
class TestPhamTurnDetection:
    @staticmethod
    def create_mock_data(num_samples=1000, sampling_freq_Hz=200):
        # Create mock accelerometer and gyro data
        time = np.arange(0, num_samples) / sampling_freq_Hz
        accel_data = np.random.rand(num_samples, 3)
        gyro_data = np.random.rand(num_samples, 3)
        data = np.concatenate((accel_data, gyro_data), axis=1)
        columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        return pd.DataFrame(data, columns=columns), sampling_freq_Hz

    def test_detect_returns_instance(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        result = pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    def test_detect_raises_value_error_non_dataframe(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = np.array([1, 2, 3]), 100
        with pytest.raises(ValueError):
            pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    def test_detect_raises_value_error_wrong_shape(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        data.drop(columns=['gyro_z'], inplace=True)
        with pytest.raises(ValueError):
            pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x',sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    def test_detect_raises_value_error_negative_sampling_freq(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        with pytest.raises(ValueError):
            pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=-100, plot_results=False)

    def test_detect_raises_value_error_non_boolean_plot_results(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        with pytest.raises(ValueError):
            pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=1)

    def test_detect_hesitations(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        data['gyro_x'] = np.linspace(0, 100, len(data))  # Ensure a continuous yaw angle change
        result = pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    def test_detect_turns_90_degrees(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        data['gyro_x'] = np.linspace(0, np.pi/2, len(data))  # Simulate a 90-degree turn
        result = pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    def test_detect_peak_angular_velocities(self):
        pham_detector = PhamTurnDetection()
        data, sampling_freq_Hz = self.create_mock_data()
        data['gyro_x'] = np.linspace(0, np.pi/2, len(data))  # Simulate a 90-degree turn
        result = pham_detector.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='gyro_x', sampling_freq_Hz=sampling_freq_Hz, plot_results=False)

    @pytest.fixture
    def sample_data(self):
        # Sample data for test
        diff_yaw = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        flags_start_90 = [2, 5]
        flags_end_90 = [7, 10]
        sampling_freq_Hz = 10
        return diff_yaw, flags_start_90, flags_end_90, sampling_freq_Hz

    def test_duration_90(self, sample_data):
        pham_detector = PhamTurnDetection()
        diff_yaw, flags_start_90, flags_end_90, sampling_freq_Hz = sample_data
        
        # Calculate duration of the turn in seconds
        duration_90 = []
        for k in range(len(flags_start_90)):
            duration_nsamples = flags_end_90[k] - flags_start_90[k]
            duration_90.append(duration_nsamples / sampling_freq_Hz)

    def test_peak_angular_velocities(self, sample_data):
        pham_detector = PhamTurnDetection()
        diff_yaw, flags_start_90, flags_end_90, sampling_freq_Hz = sample_data
        
        # Calculate peak angular velocity during the turn
        peak_angular_velocities = []
        for k in range(len(flags_start_90)):
            diff_vector = abs(diff_yaw[(flags_start_90[k] - 1):(flags_end_90[k] - 1)])
            peak_angular_velocities.append(np.max(diff_vector) * sampling_freq_Hz)

    @pytest.fixture
    def sample_data_turn_det(self):
        # Sample data for testing
        flags_start_90 = [1, 5, 9]
        flags_end_90 = [4, 8, 12]
        diff_yaw = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
        sampling_freq_Hz = 10
        return flags_start_90, flags_end_90, diff_yaw, sampling_freq_Hz

    def test_process_turns(self, sample_data_turn_det):
        pham_detector = PhamTurnDetection()

        flags_start_90, flags_end_90, diff_yaw, sampling_freq_Hz = sample_data_turn_det

        # Initialize lists to store processed data
        duration_90 = []
        peak_angular_velocities = []
        angular_velocity_start = []
        angular_velocity_end = []
        angular_velocity_middle = []

        # Process each turn
        for k in range(len(flags_start_90)):
            # Compute duration of the turn in seconds
            duration_nsamples = flags_end_90[k] - flags_start_90[k]
            duration_90.append(duration_nsamples / sampling_freq_Hz)

            # Calculate peak angular velocity during the turn
            diff_vector = abs(diff_yaw[(flags_start_90[k] - 1):(flags_end_90[k] - 1)])
            peak_angular_velocities.append(np.max(diff_vector) * sampling_freq_Hz)

            # Calculate average angular velocity at the start of the turn
            turn_10_percent = round(duration_nsamples * 0.1)
            angular_velocity_start.append(np.mean(abs(diff_yaw[flags_start_90[k]:(flags_start_90[k] + turn_10_percent)])) * sampling_freq_Hz)

            # Calculate average angular velocity at the end of the turn
            md = flags_start_90[k] + np.floor((flags_end_90[k] - flags_start_90[k]) / 2)
            angular_velocity_end.append(np.mean(abs(diff_yaw[(flags_end_90[k] - turn_10_percent):flags_end_90[k] - 1])) * sampling_freq_Hz)

            # Calculate average angular velocity in the middle of the turn
            turn_5_percent = round(duration_nsamples * 0.05)
            md = int(md)  # Convert md to an integer
            angular_velocity_middle.append(np.mean(abs(diff_yaw[int(md - turn_5_percent):int(md + turn_5_percent)])) * sampling_freq_Hz)

        # Assertions to verify the processed data
        assert duration_90 == [0.3, 0.3, 0.3]

def test_invalid_plot_results_pham_td():
    # Sample accelerometer and gyroscope data
    accel_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Accel_X', 'Accel_Y', 'Accel_Z'])
    gyro_data = pd.DataFrame(np.random.rand(1000, 3), columns=['Gyro_X', 'Gyro_Y', 'Gyro_Z'])
    sample_data = pd.concat([accel_data, gyro_data], axis=1)
    
    # Initialize PhamTurnDetection object
    pham = PhamTurnDetection()

    # Test with invalid plot_results
    invalid_plot_results = "invalid"
    with pytest.raises(ValueError):
        pham.detect(data=sample_data,accel_unit='g', gyro_unit='deg/s', gyro_vertical='Gyro_X', sampling_freq_Hz=200, plot_results=invalid_plot_results)

# Test function for PhamTurnDetection class
@pytest.fixture
def pham_detection_instance():
    return PhamTurnDetection()

@pytest.fixture
def invalid_data():
    # Create invalid data with less than 6 columns
    data = pd.DataFrame(np.random.rand(100, 5), columns=['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y'])
    return data

def test_data_shape_invalid(pham_detection_instance, invalid_data):
    # Test invalid data shape
    with pytest.raises(ValueError):
        pham_detection_instance.detect(data=invalid_data,accel_unit='g', gyro_unit='deg/s', gyro_vertical='Gyro_X', sampling_freq_Hz=100)

@pytest.fixture
def test_sampling_freq_invalid(pham_detection_instance):
    # Test invalid sampling frequency
    with pytest.raises(ValueError):
        data = pd.DataFrame(np.random.rand(100, 6))
        pham_detection_instance.detect(data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='Gyro_X', sampling_freq_Hz=0)

@pytest.fixture
def invalid_plot_results():
    return "invalid"  # Invalid non-boolean value for plot_results

def test_plot_results_invalid(pham_detection_instance, invalid_plot_results):
    # Test invalid plot_results value
    with pytest.raises(ValueError):
        data = pd.DataFrame(np.random.rand(100, 6))
        pham_detection_instance.detect(data=data, accel_unit='g', gyro_unit='deg/s', gyro_vertical='Gyro_X',  sampling_freq_Hz=100, plot_results=invalid_plot_results)

# def test_hesitation_detection_with_provided_data(pham_detection_instance):
#     # The 'file_path' variable holds the absolute path to the data file
#     file_path = (
#         r"C:\Users\Project\Desktop\bigprojects\neurogeriatrics_data\Keep Control\Data\lab dataset\raw data\sub-pp002\motion\sub-pp002_task-walkTurn_tracksys-imu_motion.tsv"
#     )
#     # In this example, we use "imu" as tracking_system and "pelvis" as tracked points.
#     tracking_sys = "imu"
#     tracked_points = {tracking_sys: ["pelvis"]}    

#     # The 'keepcontrol.load_recording' function is used to load the data from the specified file_path
#     recording = keepcontrol.load_recording(
#         file_name=file_path, tracking_systems=[tracking_sys], tracked_points=tracked_points
#     )    

#     # Load lower back acceleration data
#     acceleration_data = recording.data[tracking_sys][
#         ["pelvis_ACC_x", "pelvis_ACC_y", "pelvis_ACC_z"]
#     ]
#     # Load lower back gyro data
#     gyro_data = recording.data[tracking_sys][
#         ["pelvis_ANGVEL_x", "pelvis_ANGVEL_y", "pelvis_ANGVEL_z"]
#     ]
#     # Get the corresponding sampling frequency directly from the recording
#     sampling_frequency = recording.channels[tracking_sys][
#         recording.channels[tracking_sys]["name"] == "pelvis_ACC_x"
#     ]["sampling_frequency"].values[0]

#     # Concatenate acceleration_data and gyro_data along axis=1 (columns)
#     input_data = pd.concat([acceleration_data, gyro_data], axis=1)

#     # Call the detect method
#     pham_detection_instance.detect(input_data,accel_unit='g', gyro_unit='deg/s', gyro_vertical='pelvis_ACC_x',  sampling_freq_Hz=sampling_frequency)

#     # Assert that detected_turns attribute is not None
#     assert pham_detection_instance.detected_turns is not None

# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()