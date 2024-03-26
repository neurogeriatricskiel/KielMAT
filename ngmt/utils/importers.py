import actipy
import numpy as np
import pandas as pd
from ngmt.utils.ngmt_dataclass import NGMTRecording
from ngmt.utils.file_io import get_unit_from_type
from pathlib import Path


def import_axivity(file_path: str, tracked_point: str):
    """
    Imports Axivity data from the specified file path and constructs an NGMTRecording object.

    Args:
        file_path (str or Path): The path to the Axivity data file.
        tracked_point (str): The name of the tracked point.

    Returns:
        NGMTRecording: The NGMTRecording object containing the imported data.

    Examples:
        >>> file_path = "/path/to/axivity_data.cwa"
        >>> tracked_point = "lowerBack"
        >>> recording = import_axivity(file_path, tracked_point)
    """

    # Convert file_path to a Path object if it is a string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Read the Axivity data file and perform lowpass filtering and gravity calibration
    data, info = actipy.read_device(
        str(file_path),
        lowpass_hz=20,
        calibrate_gravity=True,
    )

    # Reset the index of the data DataFrame
    data.reset_index(inplace=True)

    # Construct the channel information

    # Set the tracked point to "lowerBack"
    tracked_point = "lowerBack"

    # Find all the columns that are named x, y or z in the data DataFrame
    accel_col_names = [col for col in data.columns if col[-1] in ["x", "y", "z"]]
    n_channels = len(accel_col_names)

    # Create the column names for the NGMTRecording object
    col_names = [f"{tracked_point}_{s}_{x}" for s in ["ACCEL"] for x in ["x", "y", "z"]]

    # Create the channel dictionary following the BIDS naming conventions
    channels_dict = {
        "name": col_names,
        "component": ["x", "y", "z"] * (n_channels // 3),
        "type": ["ACCEL"] * (n_channels),
        "tracked_point": [tracked_point] * n_channels,
        "units": ["m/s^2"] * n_channels,
        "sampling_frequency": [info["ResampleRate"]] * n_channels,
    }

    # Create the NGMTRecording object
    recording = NGMTRecording(
        data={tracked_point: data[accel_col_names]},
        channels={tracked_point: pd.DataFrame(channels_dict)},
    )

    return recording
