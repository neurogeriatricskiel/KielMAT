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

    # return an error if no tracked point is provided
    if not tracked_point:
        raise ValueError("Please provide a tracked point.")

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

    # Create the NGMTRecording object of a single tracked point
    recording = NGMTRecording(
        data={tracked_point: data[accel_col_names]},
        channels={tracked_point: pd.DataFrame(channels_dict)},
    )

    return recording


# Importher for APDM Mobility Lab system
def import_mobilityLab(
    file_name: str | Path,
    tracked_points: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imports data from an APDM Mobility Lab system from the specified file path.

    Args:
        file_name (str or Path): The absolute or relative path to the data file.
        tracked_point (str or list of str]):
            Defines for which tracked points data are to be returned.

    Returns:
        data, channels: The loaded data and channels.

    Examples:
        >>> file_path = "/path/to/sensor_data.h5"
        >>> tracked_point = "Lumbar"
        >>> recording = import_mobilityLab(file_path, tracked_point)
    """
    # Convert file_name to a Path object if it is a string
    if isinstance(file_name, str):
        file_name = Path(file_name)

    # Convert tracked_points into a list if the it is provided as a string
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]

    with h5py.File(file_name, 'r') as hfile:
        # Get monitor labels and case IDs
        monitor_labels = hfile.attrs['MonitorLabelList']
        monitor_labels = [
            s.decode("UTF-8").strip()
            for s in monitor_labels
        ]
        case_ids = hfile.attrs['CaseIdList']
        case_ids = [
            s.decode("UTF-8")[:9]
            for s in case_ids
        ]
        
        # Check if all tracked points exist in monitor labels
        for tracked_point in tracked_points:
            if tracked_point not in monitor_labels:
                print(f"Warning: Tracked point '{tracked_point}' does not exist in monitor labels.")
                # Return empty data and channels
                data = pd.DataFrame()
                channels = pd.DataFrame()

                return data, channels

        # Initialize dictionaries to store channels and data frames
        channels_dict = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": []
        }

        # Create dictionary to store data
        data_dict = {}

        # Iterate over each sensor
        for idx_sensor, (monitor_label, case_id) in enumerate(zip(monitor_labels, case_ids)):
            if monitor_label not in tracked_points:
                continue  # to next sensor name
            sample_rate = hfile[case_id].attrs['SampleRate']
            
            # Get raw data
            rawAcc = hfile[case_id]['Calibrated']['Accelerometers'][:]
            rawGyro = hfile[case_id]['Calibrated']['Gyroscopes'][:]
            rawMagn = hfile[case_id]['Calibrated']['Magnetometers'][:]

            # Populate data_dict
            data_dict[f'{monitor_label}'] = pd.DataFrame({
                f'{monitor_label}_ACCEL_x': rawAcc[:,0],
                f'{monitor_label}_ACCEL_y': rawAcc[:,1],
                f'{monitor_label}_ACCEL_z': rawAcc[:,2],
                f'{monitor_label}_GYRO_x': rawGyro[:,0],
                f'{monitor_label}_GYRO_y': rawGyro[:,1],
                f'{monitor_label}_GYRO_z': rawGyro[:,2],
                f'{monitor_label}_MAGN_x': rawMagn[:,0],
                f'{monitor_label}_MAGN_y': rawMagn[:,1],
                f'{monitor_label}_MAGN_z': rawMagn[:,2],
            })

            # Extend lists in channels_dict
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
            channels_dict["tracked_point"].extend([monitor_label] * 9)
            channels_dict["units"].extend(['m/s^2', 'm/s^2', 'm/s^2', 'rad/s', 'rad/s', 'rad/s', 'µT', 'µT', 'µT'])
            channels_dict["sampling_frequency"].extend([sample_rate] * 9)

    # Concatenate data frames from data_dict
    data = pd.concat(list(data_dict.values()), axis=1)
    
    # Create DataFrame from channels_dict
    channels = pd.DataFrame(channels_dict)
   
    return data, channels