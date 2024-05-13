import actipy
import h5py
import numpy as np
from ngmt.utils.ngmt_dataclass import NGMTRecording
from ngmt.utils.file_io import get_unit_from_type
import pandas as pd
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


def import_mobilityLab(fullFileName: str) -> NGMTRecording:
    """
    Imports data from a mobility lab system from the specified file path and constructs an NGMTRecording object.

    Args:
        file_path (str or Path): The path to the mobility lab data file.
        tracked_point (str): The name of the tracked point.

    Returns:
        NGMTRecording: The NGMTRecording object containing the imported data.

    Examples:
        >>> file_path = "/path/to/sensor_data.h5"
        >>> tracked_point = "lowerBack"
        >>> recording = import_mobilityLab(file_path, tracked_point)
    """

    # Convert file_path to a Path object if it is a string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    with h5py.File(fullFileName, 'r') as hfile:
        monitor_labels = hfile.attrs['MonitorLabelList']
        case_ids = hfile.attrs['CaseIdList']
        channels_dict = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": []
        }
        data_dict = {}
        for ixSensor in range(len(monitor_labels)):
            monitor_label = monitor_labels[ixSensor].decode('utf-8').strip()
            case_id = case_ids[ixSensor].decode('utf-8')[:9]
            sample_rate = hfile[case_id].attrs['SampleRate']
            # Raw data
            rawAcc = hfile[case_id]['Calibrated']['Accelerometers'][:]
            rawAcc = hfile[case_id]['Calibrated']['Gyroscopes'][:]
            rawMagn = hfile[case_id]['Calibrated']['Magnetometers'][:]

            data_dict[f'{monitor_label}'] = pd.DataFrame({
                f'{monitor_label}_ACCEL_x': rawAcc[:,0],
                f'{monitor_label}_ACCEL_y': rawAcc[:,1],
                f'{monitor_label}_ACCEL_z': rawAcc[:,2],
                f'{monitor_label}_GYRO_x': rawAcc[:,0],
                f'{monitor_label}_GYRO_y': rawAcc[:,1],
                f'{monitor_label}_GYRO_z': rawAcc[:,2],
                f'{monitor_label}_MAGN_x': rawMagn[:,0],
                f'{monitor_label}_MAGN_y': rawMagn[:,1],
                f'{monitor_label}_MAGN_z': rawMagn[:,2],
            })
            channels_dict["name"].extend([
                f"{monitor_label}_ACCEL_x",
                f"{monitor_label}_ACCEL_y",
                f"{monitor_label}_ACCEL_z",
                f"{monitor_label}_GYRO_x",
                f"{monitor_label}_GYRO_y",
                f"{monitor_label}_GYRO_z",
                f"{monitor_label}_MAGN_x",
                f"{monitor_label}_MAGN_y",
                f"{monitor_label}_MAGN_z",
            ])

            channels_dict["component"].extend(['x', 'y', 'z'] * 3)
            channels_dict["type"].extend(['ACCEL', 'ACCEL', 'ACCEL', 'GYRO', 'GYRO', 'GYRO', 'MAGN', 'MAGN', 'MAGN'])
            channels_dict["tracked_point"].extend([monitor_label.upper()] * 9)
            channels_dict["units"].extend(['m/s^2'] * 9)
            channels_dict["sampling_frequency"].extend([sample_rate] * 9)

    data = {monitor_label: pd.concat(list(data_dict.values()), axis=1)}
    channels = {monitor_label: pd.DataFrame(channels_dict)}

    return NGMTRecording(data=data, channels=channels)
