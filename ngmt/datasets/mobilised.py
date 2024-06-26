import numpy as np
import pandas as pd
from pathlib import Path
from pooch import DOIDownloader
from zipfile import ZipFile
from typing import Literal, Any
from ngmt.utils import matlab_loader
from ngmt.utils.ngmt_dataclass import NGMTRecording


# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type
MAP_CHANNEL_TYPES = {
    "Acc": "ACCEL",
    "Gyr": "GYRO",
    "Mag": "MAGN",
    "Bar": "BARO",
    # "Temp": "TEMP"
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


def load_file(
    tracking_systems: None | str | list[str] = None,
    tracked_points: None | dict[str, str] | dict[str, list[str]] = None,
    cohort: Literal["PFF", "PD", "MS", "HA", "COPD", "CHF"] = "PFF",
    file_name: str = "data.mat",
    dataset_path: str | Path = Path(__file__).parent / "_mobilised",
    progressbar: None | bool = None,
) -> dict[str, Any]:
    # Local reference to the available tracking systems and tracked points
    _AVAILABLE_TRACKING_SYSTEMS = ["SU", "PressureInsoles_raw", "DistanceModule_raw"]
    _AVAILABLE_TRACKED_POINTS = {
        "SU": ["LowerBack"],
        "PressureInsoles_raw": ["LeftFoot", "RightFoot"],
        "DistanceModule_raw": ["LeftFoot", "RightFoot"],
    }

    # Check the tracking systems
    tracking_systems = (
        _AVAILABLE_TRACKING_SYSTEMS if tracking_systems is None else tracking_systems
    )
    tracking_systems = (
        [tracking_systems] if isinstance(tracking_systems, str) else tracking_systems
    )
    if any(
        [track_sys not in _AVAILABLE_TRACKING_SYSTEMS for track_sys in tracking_systems]
    ):
        raise ValueError(
            f"Invalid tracking system. Available options are: {_AVAILABLE_TRACKING_SYSTEMS}"
        )

    # Set the tracked points
    tracked_points = (
        _AVAILABLE_TRACKED_POINTS if tracked_points is None else tracked_points
    )
    if any([track_sys not in tracking_systems for track_sys in tracked_points.keys()]):
        raise ValueError(
            f"You have specified tracked points for a tracking system that you have not requested (requested: {tracking_systems})."
        )
    for track_sys in tracked_points.keys():
        if isinstance(tracked_points[track_sys], str):
            tracked_points[track_sys] = [tracked_points[track_sys]] # type: ignore

    # Loop over the tracked points and check if they are valid
    for track_sys, tracked_points_list in tracked_points.items():
        if any(
            [
                tracked_point not in _AVAILABLE_TRACKED_POINTS[track_sys]
                for tracked_point in tracked_points_list
            ]
        ):
            raise ValueError(
                f"Invalid tracked point for tracking system '{track_sys}'. Available options are: {_AVAILABLE_TRACKED_POINTS[track_sys]}."
            )

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

    # Extract data for given tracking system
    recording_data = dict()
    channel_data = dict()
    for track_sys in tracking_systems:
        # Set up dictionary to store channel data
        channel_data[track_sys] = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": [],
        }

        data_arr = None
        for tracked_point in tracked_points[track_sys]:
            _data = (
                data_dict["Standards"]
                if track_sys in ["PressureInsoles_raw", "DistanceModule_raw"]
                else data_dict
            )
            for ch_type in _data[track_sys][tracked_point].keys():
                if ch_type in MAP_CHANNEL_TYPES.keys():
                    if data_arr is None:
                        data_arr = _data[track_sys][tracked_point][ch_type]
                    else:
                        data_arr = np.column_stack(
                            (data_arr, _data[track_sys][tracked_point][ch_type])
                        )
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
                        tracked_point
                        for ch_comp in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                    ]
                    channel_data[track_sys]["units"] += [
                        MAP_CHANNEL_UNITS[ch_type]
                        for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                    ]
                    channel_data[track_sys]["sampling_frequency"] += [
                        data_dict[track_sys][tracked_point]["Fs"][ch_type]
                        for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                    ]

        if data_arr is not None:
            recording_data[track_sys] = pd.DataFrame(
                data=data_arr, columns=channel_data[track_sys]["name"]
            )

        channel_data[track_sys] = pd.DataFrame(channel_data[track_sys])

    return {"a": 1, "b": 2, "c": "three"}


def load_recording(
    file_name: str | Path,
    tracking_systems: str | list[str],
    tracked_points: str | list[str] | dict[str, str] | dict[str, list[str]],
):
    """
    Load a recording from the Mobilise-D dataset.

    Args:
        file_name (str or Path ): The absolute or relative path to the data file.
        tracking_systems (str or list of str) : A string or list of strings of tracking systems for which data are to be returned.
        tracked_points (str or list of str or dict[str, str] or dict[str, list of str]) :
            Defines for which tracked points data are to be returned.
            If a string or list of strings is provided, then these will be mapped to each requested tracking system.
            If a dictionary is provided, it should map each tracking system to either a single tracked point or a list of tracked points.

    Returns:
        NGMTRecording : An instance of the NGMTRecording dataclass containing the loaded data and channels.
    """
    # Put tracking systems into a list
    if isinstance(tracking_systems, str):
        tracking_systems = [tracking_systems]

    # Tracked points will be a dictionary mapping
    # each tracking system to a list of tracked points of interest
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]
    if isinstance(tracked_points, list):
        tracked_points = {tracksys: tracked_points for tracksys in tracking_systems}
    for k, v in tracked_points.items():
        if isinstance(v, str):
            tracked_points[k] = [v]

    # Load data
    data_dict = matlab_loader.load_matlab(file_name, top_level="data")

    # Extract data for given tracking system
    recording_data = dict()
    channel_data = dict()
    for tracksys in tracking_systems:
        channel_data[tracksys] = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": [],
        }  # ['name', 'component', 'type', 'tracked_point', 'units', 'sampling_frequency']

        data_arr = None
        for tracked_point in data_dict["TimeMeasure1"]["Recording4"][tracksys].keys():
            if tracked_point in tracked_points[tracksys]:
                for ch_type in data_dict["TimeMeasure1"]["Recording4"][tracksys][
                    tracked_point
                ].keys():
                    if ch_type in MAP_CHANNEL_TYPES.keys():
                        if data_arr is None:
                            data_arr = data_dict["TimeMeasure1"]["Recording4"][
                                tracksys
                            ][tracked_point][ch_type]

                        else:
                            data_arr = np.column_stack(
                                (
                                    data_arr,
                                    data_dict["TimeMeasure1"]["Recording4"][tracksys][
                                        tracked_point
                                    ][ch_type],
                                )
                            )
                        channel_data[tracksys]["name"] += [
                            f"{tracked_point}_{MAP_CHANNEL_TYPES[ch_type]}_{ch_comp}"
                            for ch_comp in MAP_CHANNEL_COMPONENTS[ch_type]
                        ]
                        channel_data[tracksys]["type"] += [
                            MAP_CHANNEL_TYPES[ch_type]
                            for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                        ]
                        channel_data[tracksys]["component"] += [
                            ch_comp for ch_comp in MAP_CHANNEL_COMPONENTS[ch_type]
                        ]
                        channel_data[tracksys]["tracked_point"] += [
                            tracked_point
                            for ch_comp in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                        ]
                        channel_data[tracksys]["units"] += [
                            MAP_CHANNEL_UNITS[ch_type]
                            for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                        ]
                        channel_data[tracksys]["sampling_frequency"] += [
                            data_dict["TimeMeasure1"]["Recording4"][tracksys][
                                tracked_point
                            ]["Fs"][ch_type]
                            for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
                        ]

        if data_arr is not None:
            recording_data[tracksys] = pd.DataFrame(
                data=data_arr, columns=channel_data[tracksys]["name"]
            )

        channel_data[tracksys] = pd.DataFrame(channel_data[tracksys])

    return NGMTRecording(data=recording_data, channels=channel_data)
