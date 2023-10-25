import os
import numpy as np
from ..utils import matlab_loader as matlab_loader
from ..utils.ngmt_data_classes import (
    FileInfo,
    EventData,
    ChannelData,
    RecordingData,
    MotionData,
)

# Set dataset name
_DATASET_NAME = "Mobilise-D"

# Dictionary that maps sensor types to their corresponding units of measurement
_MAP_UNITS = {
    "Acc": "g",  # Accelerometer data: units of gravity ('g')
    "Gyr": "deg/s",  # Gyroscope data: degrees per second ('deg/s')
    "Mag": "microTesla",  # Magnetometer data: microteslas ('microTesla')
    "Bar": "hPa",  # Barometer data: hectopascals ('hPa')
}

_MAP_CHANNEL_NAMES = {"Acc": "ACCEL", "Gyr": "GYRO", "Mag": "MAGN", "Bar": "BARO"}


def load_file(file_path: str) -> RecordingData:
    """
    Args:
        file_path (str): Path to the data file (*.mat).

    Returns:
        MotionData: An instance of a `MotionData` object.
    """
    # Split path and filename
    path_name, file_name = os.path.split(file_path)
    sub_path_name, session_name = os.path.split(path_name)
    _, sub_id = os.path.split(sub_path_name)

    # Load data from the MATLAB file
    data_dict = matlab_loader.load_matlab(file_path, top_level="data")

    # Set file info
    file_info = FileInfo(
        SubjectID=sub_id,
        TaskName=session_name,
        DatasetName=_DATASET_NAME,
        FilePath=file_path,
    )

    # Load the data into a dictionary
    data_dict = matlab_loader.load_matlab(file_name=file_path, top_level="data")
    su_data = data_dict["TimeMeasure1"]["Recording4"]["SU"]

    # Get data from the INDIP system
    num_tracked_points = len(su_data.keys())
    data = None
    channels_dict = {
        "name": [],
        "component": [],
        "ch_type": [],
        "tracked_point": [],
        "units": [],
        "range": [],
        "sampling_frequency": [],
    }

    # Loop over the tracked points
    for i, tracked_point in zip(range(num_tracked_points), su_data.keys()):

        # Loop over the channel types
        for ch_type in su_data[tracked_point]["Fs"].keys():

            # Get the corresponding sensor readings
            readings = su_data[tracked_point][ch_type]

            # Check dimensions of data array
            if readings.ndim == 1:
                readings = np.expand_dims(readings, axis=-1)

            # Add info to channels data
            comps = ["x", "y", "z"] if ch_type in ["Acc", "Gyr", "Mag"] else ["n/a"]
            channels_dict["name"] += [
                f"{tracked_point}_{_MAP_CHANNEL_NAMES[ch_type]}_{x}"
                for x in comps
            ]
            channels_dict["component"] += comps
            channels_dict["ch_type"] += [_MAP_CHANNEL_NAMES[ch_type] for _ in range(readings.shape[-1])]
            channels_dict["tracked_point"] += [tracked_point for _ in range(readings.shape[-1])]
            channels_dict["units"] += [_MAP_UNITS[ch_type] for _ in range(readings.shape[-1])]
            channels_dict["sampling_frequency"] += [
                su_data[tracked_point]["Fs"][ch_type] for _ in range(readings.shape[-1])
            ]

            # Add current readings to cumulative data array
            if data is None:
                data = readings
            else:
                data = np.concatenate(
                    (data, readings), axis=1
                )

    # Generate ChannelData object
    channel_data = ChannelData(
        name=channels_dict["name"],
        component=channels_dict["component"],
        ch_type=channels_dict["ch_type"],
        tracked_point=channels_dict["tracked_point"],
        units=channels_dict["units"],
        sampling_frequency=channels_dict["sampling_frequency"][0],
    )

    # Generate RecordingData object
    recording_data = RecordingData(
        name=f"{_DATASET_NAME}_TVS",
        data=data,
        sampling_frequency=channels_dict["sampling_frequency"][0],
        times=su_data[tracked_point]["Timestamp"],
        channels=channel_data,
        start_time=su_data[tracked_point]["Timestamp"][0],
    )

    return MotionData(
        data=[recording_data],
        times=recording_data.times,
        info=[file_info],
    )
