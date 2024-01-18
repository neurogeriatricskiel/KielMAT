import numpy as np
import pandas as pd
import pathlib
from ngmt.utils import matlab_loader
from ngmt.utils.data_classes import NGMTRecording


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


def load_recording(
    file_name: str | pathlib.Path,
    tracking_systems: str | list[str],
    tracked_points: str | list[str] | dict[str, str] | dict[str, list[str]],
):
    """Load a recording from the Mobilise-D dataset.

    Parameters
    ----------
    file_name : str | pathlib.Path
        The absolute or relative path to the data file.
    tracking_systems : str | list[str]
        A string or list of strings of tracking systems for which data are to be returned.
    tracked_points : str | list[str] | dict[str, str] | dict[str, list[str]]
        Defines for which tracked points data are to be returned.
        If a string or list of strings is provided, then these will be mapped to each requested tracking system.

    Returns
    -------
    _ : NGMTRecording
        An instance of the NGMTRecording dataclass.
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
            "type": [],
            "component": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": [],
        }

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
                            ch_type for _ in range(len(MAP_CHANNEL_COMPONENTS[ch_type]))
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
