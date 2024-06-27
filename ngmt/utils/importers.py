import actipy
import h5py
import numpy as np
import pandas as pd
from ngmt.utils.ngmt_dataclass import NGMTRecording
from ngmt.utils.file_io import get_unit_from_type
from pathlib import Path
from typing import Union

# Importher for Axivity
def import_axivity(file_path: str, tracked_point: str):
    """
    Imports Axivity data from the specified file path and 
    return the data and channel formatted to be used in a NGMTRecording object.

    Args:
        file_path (str or Path): The path to the Axivity data file.
        tracked_point (str): The name of the tracked point.

    Returns:
        dict, dict: The loaded data and channels as dictionaries.

    Examples:
        >>> file_path = "/path/to/axivity_data.cwa"
        >>> tracked_point = "lowerBack"
        >>> data, channels = import_axivity(file_path, tracked_point)
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
    channels = {
        "name": col_names,
        "component": ["x", "y", "z"] * (n_channels // 3),
        "type": ["ACCEL"] * (n_channels),
        "tracked_point": [tracked_point] * n_channels,
        "units": ["m/s^2"] * n_channels,
        "sampling_frequency": [info["ResampleRate"]] * n_channels,
    }

    return data, channels


# Importher for APDM Mobility Lab for different versions
def import_apdm_mobilitylab(
    file_name: str | Path, 
    tracked_points: str | list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imports data from an APDM Mobility Lab system's different versions from the specified file path.

    Args:
        file_name (str or Path): The absolute or relative path to the data file.
        tracked_points (str or list of str]): Defines for which tracked points data are to be returned.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The loaded data and channels as dataframes.

    Examples:
        >>> file_path = "/path/to/sensor_data.h5"
        >>> tracked_point = "Lumbar"
        >>> data, channels = import_mobilityLab_all(file_path, tracked_point)
    """
    # Convert file_name to a Path object if it is a string
    if isinstance(file_name, str):
        file_name = Path(file_name)

    # Convert tracked_points into a list if it is provided as a string
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]

    # Mapping of sensor types to make them consistent with NGMT dataclass definition
    sensor_type_mapping = {
        'Accelerometer': 'ACCEL',
        'Gyroscope': 'GYRO',
        'Magnetometer': 'MAGN',
        'Accelerometers': 'ACCEL',
        'Gyroscopes': 'GYRO',
        'Magnetometers': 'MAGN'
    }

    with h5py.File(file_name, 'r') as hfile:
        # Check if there is an attribute or dataset that indicates the version
        if 'FileFormatVersion' in hfile.attrs:
            version = hfile.attrs['FileFormatVersion']
        else:
            raise ValueError("Version attribute not found in the h5 file.")

        # Initialize dictionaries to store channels and data frames
        channels_dict = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": []
        }
        data_dict = {}

        # Check the version
        if version == 5:
            sensors_group = hfile['Sensors']

            # Structure for version 5
            monitor_labels = list(sensors_group.keys())
            sensor_to_label = {
                sensor_id: sensors_group[sensor_id]['Configuration'].attrs['Label 0'].decode('utf-8')
                for sensor_id in monitor_labels
            }

            # Convert tracked_points to sensor IDs using sensor_to_label mapping
            tracked_points = [sensor for sensor, label in sensor_to_label.items() if label in tracked_points]

            # Track invalid tracked points
            invalid_tracked_points = [tp for tp in tracked_points if tp not in monitor_labels]

            if invalid_tracked_points:
                raise ValueError(f"The following tracked points do not exist in sensor names: {invalid_tracked_points}")

            # Iterate over each sensor
            for sensor_name in monitor_labels:
                if sensor_name not in tracked_points:
                    continue  # to next sensor name

                sensor_data = sensors_group[sensor_name]
                sample_rate = sensor_data['Configuration'].attrs['Sample Rate']
                label = sensor_to_label[sensor_name]

                # Extract and append sensor data to the DataFrame
                for axis_label in ['x', 'y', 'z']:
                    for sensor_type in ['Accelerometer', 'Gyroscope', 'Magnetometer']:
                        column_name = f"{label}_{sensor_type_mapping[sensor_type]}_{axis_label}"
                        if sensor_type in sensor_data:
                            raw_data = sensor_data[sensor_type][:]
                            data_dict[column_name] = raw_data[:, 'xyz'.index(axis_label)]

                            # Extend lists in channels_dict
                            channels_dict["name"].append(column_name)
                            channels_dict["component"].append(axis_label)
                            channels_dict["type"].append(sensor_type_mapping[sensor_type])
                            channels_dict["tracked_point"].append(label)
                            channels_dict["units"].append(sensor_data[sensor_type].attrs['Units'].decode())
                            channels_dict["sampling_frequency"].append(sample_rate)

        else:
            # Structure for version 3 and 4
            monitor_labels = hfile.attrs['MonitorLabelList']
            monitor_labels = [s.decode("UTF-8").strip() for s in monitor_labels]
            case_ids = hfile.attrs['CaseIdList']
            case_ids = [s.decode("UTF-8")[:9] for s in case_ids]

            # Track invalid tracked points
            invalid_tracked_points = [tp for tp in tracked_points if tp not in monitor_labels]

            if invalid_tracked_points:
                raise ValueError(f"The following tracked points do not exist in monitor labels: {invalid_tracked_points}")

            # Iterate over each sensor
            for idx_sensor, (monitor_label, case_id) in enumerate(zip(monitor_labels, case_ids)):
                if monitor_label not in tracked_points:
                    continue  # Skip to next sensor name

                sample_rate = hfile[case_id].attrs['SampleRate']
                sensor_data = hfile[case_id]['Calibrated']
                
                # Extract data for Accelerometers, Gyroscopes, and Magnetometers
                sensor_types = ['Accelerometers', 'Gyroscopes', 'Magnetometers']
                for sensor_type in sensor_types:
                    if sensor_type in sensor_data:
                        raw_data = sensor_data[sensor_type][:]
                        units = sensor_data[sensor_type].attrs['Units'].decode()
                        
                        for axis_label in ['x', 'y', 'z']:
                            column_name = f"{monitor_label}_{sensor_type_mapping[sensor_type]}_{axis_label}"
                            data_dict[column_name] = raw_data[:, 'xyz'.index(axis_label)]

                            # Extend lists in channels_dict
                            channels_dict["name"].append(column_name)
                            channels_dict["component"].append(axis_label)
                            channels_dict["type"].append(sensor_type_mapping[sensor_type])
                            channels_dict["tracked_point"].append(monitor_label)
                            channels_dict["units"].append(units)
                            channels_dict["sampling_frequency"].append(sample_rate)

    # Create DataFrame from data_dict
    data = pd.DataFrame(data_dict)
    
    # Create DataFrame from channels_dict
    channels = pd.DataFrame(channels_dict)

    return data, channels
