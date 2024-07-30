import numpy as np
import pandas as pd
from pathlib import Path
from pooch import DOIDownloader
from zipfile import ZipFile
from typing import Literal, Any
from kielmat.utils import matlab_loader
from kielmat.utils.kielmat_dataclass import KielMATRecording


# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type
MAP_CHANNEL_TYPES = {
    "Acc": "ACCEL",
    "Gyr": "GYRO",
    "Mag": "MAGN",
    "Bar": "BARO",
    # "Temp": "TEMP",
}

MAP_CHANNEL_COMPONENTS = {
    "Acc": ["x", "y", "z"],
    "Gyr": ["x", "y", "z"],
    "Mag": ["x", "y", "z"],
    "Bar": ["n/a"],
}

# See: https://www.nature.com/articles/s41597-023-01930-9
MAP_CHANNEL_UNITS = {
    "Acc": "g",
    "Gyr": "deg/s",
    "Mag": "µT",
    "Bar": "hPa",  # "Temp": "deg C"
}


def fetch_dataset(
    progressbar: bool = True,
    dataset_path: str | Path = Path(__file__).parent / "_mobilised",
) -> None:
    """Fetch the Mobilise-D dataset from the Zenodo repository.

    Args:
        progressbar (bool, optional): Whether to display a progressbar. Defaults to True.
        dataset_path (str | Path, optional): The path where the dataset is stored. Defaults to Path(__file__).parent/"_mobilised".
    """
    dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path

    # Check if zip archive has already been downloaded
    if not dataset_path.exists():
        dataset_path.parent.joinpath("_mobilised").mkdir(parents=True, exist_ok=True)
    _output_file = dataset_path.joinpath("Mobilise-D_dataset.zip")

    if not _output_file.exists():
        # Set the URL to the dataset
        _url = "doi:10.5281/zenodo.7547125/Mobilise-D dataset_1-18-2023.zip"

        # Instantiate a downloader object
        downloader = DOIDownloader(progressbar=progressbar)
        downloader(url=_url, output_file=_output_file, pooch=None)

    # Extract the dataset
    with ZipFile(_output_file, "r") as zip_ref:
        zip_ref.extractall(dataset_path)
    return


def load_recording(
    cohort: Literal["PFF", "PD", "MS", "HA", "COPD", "CHF"] = "PFF",
    file_name: str = "data.mat",
    dataset_path: str | Path = Path(__file__).parent / "_mobilised",
    progressbar: bool = True,
) -> KielMATRecording:
    """Load a recording from the Mobilise-D dataset.

    If the dataset has not yet been downloaded, then is fetched from the Zenodo repository using the pooch package.

    Args:
        cohort (Literal["PFF", "PD", "MS", "HA", "COPD", "CHF"], optional): The cohort from which data should be loaded. Defaults to "PFF".
        file_name (str, optional): The filename of the data file. Defaults to "data.mat".
        dataset_path (str | Path, optional): The path to the dataset. Defaults to Path(__file__).parent/"_mobilised".
        progressbar (bool, optional): Whether to display a progressbar when fetching the data. Defaults to True.

    Returns:
        KielMATRecording: An instance of the KielMATRecording dataclass containing the loaded data and channels.
    """

    # Fetch the dataset if it does not exist
    progressbar = False if not progressbar else progressbar
    file_path = Path(dataset_path) / cohort / file_name
    if not file_path.exists():
        fetch_dataset(progressbar=progressbar, dataset_path=dataset_path)

    # Load the data from the file path
    data_dict = matlab_loader.load_matlab(file_path, top_level="data")
    data_dict = data_dict["TimeMeasure1"][
        "Recording4"
    ]  # to simplify the data structure

    # Get the data into a numpy ndarray
    track_sys = "SU"
    recording_data = {"SU": None}
    channel_data = {
        "SU": {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": [],
        }
    }
    for tracked_point in data_dict[track_sys].keys():
        for ch_type in data_dict[track_sys][tracked_point].keys():
            if ch_type not in MAP_CHANNEL_TYPES.keys():
                continue  # to next channel type

            # Accumulate the data
            if recording_data[track_sys] is None:
                recording_data[track_sys] = data_dict[track_sys][tracked_point][ch_type]
            else:
                recording_data[track_sys] = np.column_stack(
                    (recording_data[track_sys], data_dict[track_sys][tracked_point][ch_type])  # type: ignore
                )  # type: ignore

            # Accumulate the channel data
            channel_data[track_sys]["name"] += [
                f"{tracked_point}_{MAP_CHANNEL_TYPES[ch_type]}_{ch_comp}"
                for ch_comp in MAP_CHANNEL_COMPONENTS[ch_type]
            ]
            channel_data[track_sys]["type"] += [
                MAP_CHANNEL_TYPES[ch_type]
                for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
            ]
            channel_data[track_sys]["component"] += [
                ch_comp for ch_comp in MAP_CHANNEL_COMPONENTS[ch_type]
            ]
            channel_data[track_sys]["tracked_point"] += [
                tracked_point for ch_comp in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
            ]
            channel_data[track_sys]["units"] += [
                MAP_CHANNEL_UNITS[ch_type]
                for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
            ]
            channel_data[track_sys]["sampling_frequency"] += [
                data_dict[track_sys][tracked_point]["Fs"][ch_type]
                for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
            ]

    return KielMATRecording(
        data={
            track_sys: pd.DataFrame(
                data=recording_data[track_sys], columns=channel_data[track_sys]["name"]
            )
        },
        channels={track_sys: pd.DataFrame(channel_data[track_sys])},
    )
