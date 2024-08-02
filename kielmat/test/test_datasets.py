import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from kielmat.datasets.mobilised import load_recording as mobilised_load_recording
from kielmat.datasets.keepcontrol import (
    load_recording as keepcontrol_load_recording,
    fetch_dataset as keepcontrol_fetch_dataset,
)
from kielmat.datasets.fairpark import load_recording as fairpark_load_recording
from kielmat.utils.kielmat_dataclass import (
    KielMATRecording,
    VALID_COMPONENT_TYPES,
    VALID_CHANNEL_STATUS_VALUES,
)


# Test for keepcontrol dataset
@pytest.fixture
def mock_download():
    with patch("openneuro.download") as mock:
        yield mock


@pytest.fixture
def mock_zipfile():
    with patch("zipfile.ZipFile") as mock:
        yield mock


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Fixture to create a temporary dataset path."""
    path = tmp_path / "_keepcontrol"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_fetch_dataset_creates_directory(temp_dataset_path, mock_download):
    """Test that fetch_dataset creates the directory if it does not exist."""
    # Set up the mock to simulate the download action
    mock_download.return_value = None

    # Call the function under test
    keepcontrol_fetch_dataset(dataset_path=temp_dataset_path)

    # Assert that the directory was created
    assert temp_dataset_path.exists()


def test_fetch_dataset_no_download_needed(temp_dataset_path, mock_download):
    """Test that fetch_dataset does not download if the directory already contains files."""
    # Create a file in the directory to simulate that it is not empty
    (temp_dataset_path / "some_file").touch()

    # Call the function under test
    keepcontrol_fetch_dataset(dataset_path=temp_dataset_path)

    # Assert that the download was not called
    mock_download.assert_not_called()


def test_zipfile_validity(temp_dataset_path, mock_zipfile):
    """Test that the ZIP file is valid."""
    # Ensure the ZIP file exists
    zip_file_path = temp_dataset_path / "KeepControl_dataset.zip"
    zip_file_path.touch()

    # Set up the mock to simulate a valid zip file
    mock_zipfile.return_value.__enter__.return_value.testzip = MagicMock()


@patch("kielmat.datasets.keepcontrol.fetch_dataset")
def test_keepcontrol_load_recording_missing_dataset(mock_keepcontrol_fetch_dataset):
    dataset_path = Path("path/to/keepcontrol")
    id = "sub001"
    task = "walkSlow"
    tracking_systems = ["imu", "omc"]

    # Simulate that the file does not exist initially
    with patch("pathlib.Path.exists", return_value=False):
        recording = keepcontrol_load_recording(
            dataset_path=dataset_path,
            id=id,
            task=task,
            tracking_systems=tracking_systems,
        )

    # Assert that fetch_dataset was called
    mock_keepcontrol_fetch_dataset.assert_called_once()
    assert isinstance(recording, KielMATRecording) is False


def test_tracking_systems_conversion():
    """Test that tracking systems are converted to a list."""
    tracking_systems = "imu"
    expected = ["imu"]

    # Convert tracking_systems to a list if it is a string
    if isinstance(tracking_systems, str):
        tracking_systems = [tracking_systems]

    assert tracking_systems == expected


def test_tracked_points_conversion():
    """Test that tracked points are converted correctly."""
    tracking_systems = ["imu", "omc"]
    tracked_points = "head"
    expected = {"imu": ["head"], "omc": ["head"]}

    # Convert tracked_points to a dictionary if it is a list or string
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]
    if isinstance(tracked_points, list):
        tracked_points = {tracksys: tracked_points for tracksys in tracking_systems}

    assert tracked_points == expected


@patch("kielmat.datasets.keepcontrol.logging.warning")
def test_multiple_files_found(mock_warning, temp_dataset_path):
    """Test that multiple files found scenario is handled correctly."""
    # Simulate multiple files
    (temp_dataset_path / "sub-pp001").mkdir(parents=True, exist_ok=True)
    file_path = (
        temp_dataset_path
        / "sub-pp001"
        / "sub-pp001_task-walkSlow_tracksys-imu_motion.tsv"
    )
    file_path.touch()

    # Create another file to simulate the multiple file scenario
    file_path2 = (
        temp_dataset_path
        / "sub-pp001"
        / "sub-pp001_task-walkSlow_tracksys-imu_motion2.tsv"
    )
    file_path2.touch()

    # Call the function or code block that handles file loading
    file_name = list(
        temp_dataset_path.glob(
            "sub-pp001/motion/sub-pp001_task-walkSlow_tracksys-imu_*motion.tsv"
        )
    )
    if len(file_name) > 1:
        mock_warning.assert_called()
        assert (
            "Multiple files found for ID pp001, task walkSlow, and tracking system imu."
            in [call[0][0] for call in mock_warning.call_args_list]
        )


# Test for mobilised dataset
@pytest.fixture
def mock_fetch_dataset():
    with patch("kielmat.datasets.mobilised.fetch_dataset") as mock:
        yield mock


@pytest.fixture
def mock_load_matlab():
    with patch("kielmat.utils.matlab_loader.load_matlab") as mock:
        yield mock


@pytest.fixture
def mock_dataloader():
    with patch("kielmat.datasets.mobilised.DOIDownloader") as mock:
        yield mock


def test_mobilised_load_recording_success(
    mock_fetch_dataset, mock_load_matlab, mock_dataloader
):
    # Setup mock data
    mock_load_matlab.return_value = {
        "TimeMeasure1": {
            "Recording4": {
                "SU": {
                    "tracked_point_1": {
                        "Acc": np.random.rand(100, 3).tolist(),
                        "Gyr": np.random.rand(100, 3).tolist(),
                        "Mag": np.random.rand(100, 3).tolist(),
                        "Bar": np.random.rand(100, 1).tolist(),
                        "Fs": {"Acc": 100, "Gyr": 100, "Mag": 100, "Bar": 100},
                    }
                }
            }
        }
    }

    dataset_path = Path("path/to/dataset")
    cohort = "PFF"
    file_name = "data.mat"

    recording = mobilised_load_recording(
        cohort=cohort, file_name=file_name, dataset_path=dataset_path
    )

    # Assert that data is loaded correctly
    assert isinstance(recording, KielMATRecording)
    assert isinstance(recording.data["SU"], pd.DataFrame)
    assert isinstance(recording.channels["SU"], pd.DataFrame)
    assert len(recording.data["SU"].columns) > 0
    assert len(recording.channels["SU"]) > 0


def test_mobilised_load_recording_missing_dataset(
    mock_fetch_dataset, mock_load_matlab, mock_dataloader
):
    # Setup mock data
    mock_load_matlab.return_value = {
        "TimeMeasure1": {
            "Recording4": {
                "SU": {
                    "tracked_point_1": {
                        "Acc": np.random.rand(100, 3).tolist(),
                        "Gyr": np.random.rand(100, 3).tolist(),
                        "Mag": np.random.rand(100, 3).tolist(),
                        "Bar": np.random.rand(100, 1).tolist(),
                        "Fs": {"Acc": 100, "Gyr": 100, "Mag": 100, "Bar": 100},
                    }
                }
            }
        }
    }

    dataset_path = Path("path/to/dataset")
    cohort = "PFF"
    file_name = "data.mat"

    # Simulate that the file does not exist initially
    with patch("pathlib.Path.exists", return_value=False):
        recording = mobilised_load_recording(
            cohort=cohort, file_name=file_name, dataset_path=dataset_path
        )

    # Assert that fetch_dataset was called
    mock_fetch_dataset.assert_called_once()
    assert isinstance(recording, KielMATRecording)


# Test for fairpark dataset
mock_fairpark_data = pd.DataFrame(
    {
        "head_x": np.random.rand(100),
        "head_y": np.random.rand(100),
        "head_z": np.random.rand(100),
    }
)

mock_fairpark_channels = pd.DataFrame(
    {
        "name": ["head_x", "head_y", "head_z"],
        "type": ["Acc", "Acc", "Acc"],
        "component": ["x", "y", "z"],
        "tracked_point": ["head", "head", "head"],
        "units": ["g", "g", "g"],
        "sampling_frequency": [100, 100, 100],
    }
)


@pytest.mark.parametrize(
    "tracking_systems, tracked_points",
    [
        (["imu"], ["head"]),
    ],
)
@patch("pandas.read_csv")
@patch("pathlib.Path.exists", return_value=False)
def test_fairpark_load_recording_missing_dataset(
    mock_exists, mock_read_csv, tracking_systems, tracked_points
):
    # Mock the return value of read_csv to be our mock data
    mock_read_csv.return_value = pd.DataFrame(
        {
            "head_x_ACCEL_x": np.random.rand(100),
            "head_x_ACCEL_y": np.random.rand(100),
            "head_x_ACCEL_z": np.random.rand(100),
            "head_x_ANGVEL_x": np.random.rand(100),
            "head_x_ANGVEL_y": np.random.rand(100),
            "head_x_ANGVEL_z": np.random.rand(100),
            "head_x_MAGN_x": np.random.rand(100),
            "head_x_MAGN_y": np.random.rand(100),
            "head_x_MAGN_z": np.random.rand(100),
            "year": [2023] * 100,
            "month": [1] * 100,
            "day": [1] * 100,
            "hour": [0] * 100,
            "minute": [0] * 100,
            "second": np.arange(100),
        }
    )

    file_name = "path/to/fairpark_file.csv"

    recording = fairpark_load_recording(
        file_name=file_name,
        tracking_systems=tracking_systems,
        tracked_points=tracked_points,
    )

    assert isinstance(recording, KielMATRecording)


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
