import actipy
import h5py
import numpy as np
from kielmat.utils.kielmat_dataclass import KielMATRecording
from kielmat.utils.file_io import get_unit_from_type
import pandas as pd
from pathlib import Path


def import_axivity(
    file_path: str, tracked_point: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imports and processes data from an Axivity device file.

    Args:
        file_path (str): Path to the Axivity data file.
        tracked_point (str): Label for the tracked body point (e.g., 'wrist').

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - data (pd.DataFrame): Processed accelerometer data with time information.
            - channels (pd.DataFrame): Channel information DataFrame with metadata such as
              component, type, units, and sampling frequency.

    Raises:
        ValueError: If no tracked point is provided.
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

    # Create the column names for the KielMATRecording object
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

    # Convert channels dictionary to a DataFrame
    channels = pd.DataFrame(channels_dict)

    return data, channels


# Importher for APDM Mobility Lab system
def import_mobilityLab(
    file_name: str | Path,
    tracked_points: str | list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imports and processes data from an APDM Mobility Lab system file.

    Args:
        file_name (str | Path): Path to the Mobility Lab HDF5 file.
        tracked_points (str | list[str]): Name or list of names for tracked body points to import.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - data (pd.DataFrame): DataFrame with combined accelerometer, gyroscope,
              and magnetometer data for each tracked point.
            - channels (pd.DataFrame): DataFrame containing metadata, including sensor name,
              component, type, units, and sampling frequency.

    Raises:
        ValueError: If any specified tracked point does not exist in the file's monitor labels.
    """
    # Convert file_name to a Path object if it is a string
    if isinstance(file_name, str):
        file_name = Path(file_name)

    # Convert tracked_points into a list if the it is provided as a string
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]

    with h5py.File(file_name, "r") as hfile:
        # Get monitor labels and case IDs
        monitor_labels = hfile.attrs["MonitorLabelList"]
        monitor_labels = [s.decode("UTF-8").strip() for s in monitor_labels]
        case_ids = hfile.attrs["CaseIdList"]
        case_ids = [s.decode("UTF-8")[:9] for s in case_ids]

        # Track invalid tracked points
        invalid_tracked_points = [
            tp for tp in tracked_points if tp not in monitor_labels
        ]

        if invalid_tracked_points:
            raise ValueError(
                f"The following tracked points do not exist in monitor labels: {invalid_tracked_points}"
            )

        # Initialize dictionaries to store channels and data frames
        channels_dict = {
            "name": [],
            "component": [],
            "type": [],
            "tracked_point": [],
            "units": [],
            "sampling_frequency": [],
        }

        # Create dictionary to store data
        data_dict = {}

        # Iterate over each sensor
        for idx_sensor, (monitor_label, case_id) in enumerate(
            zip(monitor_labels, case_ids)
        ):
            if monitor_label not in tracked_points:
                continue  # to next sensor name
            sample_rate = hfile[case_id].attrs["SampleRate"]

            # Get raw data
            rawAcc = hfile[case_id]["Calibrated"]["Accelerometers"][:]
            rawGyro = hfile[case_id]["Calibrated"]["Gyroscopes"][:]
            rawMagn = hfile[case_id]["Calibrated"]["Magnetometers"][:]

            # Populate data_dict
            data_dict[f"{monitor_label}"] = pd.DataFrame(
                {
                    f"{monitor_label}_ACCEL_x": rawAcc[:, 0],
                    f"{monitor_label}_ACCEL_y": rawAcc[:, 1],
                    f"{monitor_label}_ACCEL_z": rawAcc[:, 2],
                    f"{monitor_label}_GYRO_x": rawGyro[:, 0],
                    f"{monitor_label}_GYRO_y": rawGyro[:, 1],
                    f"{monitor_label}_GYRO_z": rawGyro[:, 2],
                    f"{monitor_label}_MAGN_x": rawMagn[:, 0],
                    f"{monitor_label}_MAGN_y": rawMagn[:, 1],
                    f"{monitor_label}_MAGN_z": rawMagn[:, 2],
                }
            )

            # Extend lists in channels_dict
            channels_dict["name"].extend(
                [
                    f"{monitor_label}_ACCEL_x",
                    f"{monitor_label}_ACCEL_y",
                    f"{monitor_label}_ACCEL_z",
                    f"{monitor_label}_GYRO_x",
                    f"{monitor_label}_GYRO_y",
                    f"{monitor_label}_GYRO_z",
                    f"{monitor_label}_MAGN_x",
                    f"{monitor_label}_MAGN_y",
                    f"{monitor_label}_MAGN_z",
                ]
            )

            channels_dict["component"].extend(["x", "y", "z"] * 3)
            channels_dict["type"].extend(
                [
                    "ACCEL",
                    "ACCEL",
                    "ACCEL",
                    "GYRO",
                    "GYRO",
                    "GYRO",
                    "MAGN",
                    "MAGN",
                    "MAGN",
                ]
            )
            channels_dict["tracked_point"].extend([monitor_label] * 9)
            channels_dict["units"].extend(
                ["m/s^2", "m/s^2", "m/s^2", "rad/s", "rad/s", "rad/s", "µT", "µT", "µT"]
            )
            channels_dict["sampling_frequency"].extend([sample_rate] * 9)

    # Concatenate data frames from data_dict
    data = pd.concat(list(data_dict.values()), axis=1)

    # Create DataFrame from channels_dict
    channels = pd.DataFrame(channels_dict)

    return data, channels
