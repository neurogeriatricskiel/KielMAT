import pytest
import zipfile
from kielmat.datasets import keepcontrol, mobilised
from pathlib import Path

@pytest.fixture
def ensure_mobilised_dataset():
    """Fixture to ensure the Mobilise-D dataset is correctly downloaded and is a valid ZIP file."""
    dataset_path = Path(__file__).parent / "_mobilised"
    zip_file_path = dataset_path / "Mobilise-D_dataset.zip"
    
    if not zip_file_path.exists():
        mobilised.fetch_dataset(dataset_path=dataset_path)  # Ensure dataset is fetched
    
    if not zip_file_path.exists():
        raise FileNotFoundError(f"The file {zip_file_path} was not downloaded.")
    
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.testzip()  # Test if the ZIP file is valid
    except zipfile.BadZipFile:
        raise RuntimeError(f"The file {zip_file_path} is not a valid zip file.")

@pytest.fixture
def ensure_keepcontrol_dataset():
    """Fixture to ensure the KeepControl dataset is correctly downloaded and is a valid ZIP file."""
    dataset_path = Path(__file__).parent / "_keepcontrol"
    zip_file_path = dataset_path / "KeepControl_dataset.zip"
    
    if not zip_file_path.exists():
        keepcontrol.fetch_dataset(dataset_path=dataset_path)  # Ensure dataset is fetched
    
    if not zip_file_path.exists():
        raise FileNotFoundError(f"The file {zip_file_path} was not downloaded.")
    
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.testzip()  # Test if the ZIP file is valid
    except zipfile.BadZipFile:
        raise RuntimeError(f"The file {zip_file_path} is not a valid zip file.")

# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()
