import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import zipfile
import shutil
from kielmat.datasets import keepcontrol, mobilised
from kielmat.datasets.mobilised import load_recording, fetch_dataset
from kielmat.utils.kielmat_dataclass import KielMATRecording


# Mock external dependencies
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
    keepcontrol.fetch_dataset(dataset_path=temp_dataset_path)

    # Assert that the directory was created
    assert temp_dataset_path.exists()


def test_fetch_dataset_downloads_data(temp_dataset_path, mock_download):
    """Test that fetch_dataset downloads data if the directory is empty."""
    # Ensure the directory is empty
    for item in temp_dataset_path.iterdir():
        item.unlink()

    # Set up the mock to simulate the download action
    mock_download.return_value = None

    # Call the function under test
    keepcontrol.fetch_dataset(dataset_path=temp_dataset_path)

    # Assert that the download was called
    mock_download.assert_called_once_with(
        dataset="ds005258",
        target_dir=temp_dataset_path,
    )


def test_fetch_dataset_no_download_needed(temp_dataset_path, mock_download):
    """Test that fetch_dataset does not download if the directory already contains files."""
    # Create a file in the directory to simulate that it is not empty
    (temp_dataset_path / "some_file").touch()

    # Call the function under test
    keepcontrol.fetch_dataset(dataset_path=temp_dataset_path)

    # Assert that the download was not called
    mock_download.assert_not_called()


def test_zipfile_validity(temp_dataset_path, mock_zipfile):
    """Test that the ZIP file is valid."""
    # Ensure the ZIP file exists
    zip_file_path = temp_dataset_path / "KeepControl_dataset.zip"
    zip_file_path.touch()

    # Set up the mock to simulate a valid zip file
    mock_zipfile.return_value.__enter__.return_value.testzip = MagicMock()

    # Call the function under test
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.testzip()
    except zipfile.BadZipFile:
        pytest.fail(f"The file {zip_file_path} is not a valid zip file.")


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


def test_load_recording_success(mock_fetch_dataset, mock_load_matlab, mock_dataloader):
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

    recording = load_recording(
        cohort=cohort, file_name=file_name, dataset_path=dataset_path
    )

    # Assert that data is loaded correctly
    assert isinstance(recording, KielMATRecording)
    assert isinstance(recording.data["SU"], pd.DataFrame)
    assert isinstance(recording.channels["SU"], pd.DataFrame)
    assert len(recording.data["SU"].columns) > 0
    assert len(recording.channels["SU"]) > 0


def test_load_recording_missing_dataset(
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
        recording = load_recording(
            cohort=cohort, file_name=file_name, dataset_path=dataset_path
        )

    # Assert that fetch_dataset was called
    mock_fetch_dataset.assert_called_once()
    assert isinstance(recording, KielMATRecording)


# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
