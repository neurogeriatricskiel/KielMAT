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

_MAP_CHANNEL_NAMES = {"Acc": "ACCEL", "Gyr": "GYRO", "Mag": "MAGN"}


def load_file(file_path: str) -> RecordingData:
    """
    Args:
        file_name (str): _description_

    Returns:
        IMUDataset: _description_
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
    indip_data = data_dict["TimeMeasure1"]["Recording4"]["SU_INDIP"]

    # Get data from the INDIP system
    num_tracked_points = len(indip_data.keys())
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
    for i, tracked_point in zip(range(num_tracked_points), indip_data.keys()):
        # Loop over the channel types
        for ch_type in indip_data[tracked_point]["Fs"].keys():
            # Add info to channels data
            channels_dict["name"] += [
                f"{tracked_point}_{_MAP_CHANNEL_NAMES[ch_type]}_{x}"
                for x in ["x", "y" "z"]
            ]
            channels_dict["component"] += ["x", "y", "z"]
            channels_dict["ch_type"] += [_MAP_CHANNEL_NAMES[ch_type] for _ in range(3)]
            channels_dict["tracked_point"] += [tracked_point for _ in range(3)]
            channels_dict["units"] += [_MAP_UNITS[ch_type] for _ in range(3)]
            channels_dict["sampling_frequency"] += [
                indip_data[tracked_point]["Fs"][ch_type] for _ in range(3)
            ]

            if data is None:
                data = indip_data[tracked_point][ch_type]
            else:
                data = np.concatenate(
                    (data, indip_data[tracked_point][ch_type]), axis=1
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
        times=indip_data[tracked_point]["Timestamp"],
        channels=channel_data,
        start_time=indip_data[tracked_point]["Timestamp"][0],
    )

    return MotionData(
        data=[recording_data],
        times=recording_data.times,
        info=[file_info],
    )
