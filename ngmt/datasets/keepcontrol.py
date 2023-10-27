import numpy as np
import pandas as pd
import os
import sys
from ..utils.ngmt_data_classes import (
    FileInfo,
    ChannelData,
    EventData,
    RecordingData,
    MotionData,
)

_MAP_CHANNEL_TYPES = {"ACC": "ACCEL", "ANGVEL": "GYRO", "MAGN": "MAGN"}

if sys.platform == "linux":
    _BASE_PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
elif sys.platform == "win32":
    _BASE_PATH = "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
else:
    raise OSError(
        "Currently only Windows- or Linux-based operating systems are supported!"
    )


def load_file(file_path: str) -> MotionData:
    """
    Args:
        file_path (str): Path to the data file (*.csv).

    Returns:
        MotionData: An instance of a `MotionData` object.
    """
    # Split path and file name
    path_name, file_name = os.path.split(file_path)

    # Get relevant infos
    s = file_name.replace("sub-", "")
    sub_id = s[: s.find("_task")]
    s = s[s.find("_task") + len("_task") + 1 :]
    if "_run" in file_name:
        run_name = s[: s.find("_run")]
        s = s[s.find("_run") + len("_run") + 1 :]
    task_name = s[: s.find("_tracksys")]
    s = s[s.find("_tracksys") + len("_tracksys") + 1 :]
    tracksys = s[: s.find("_")]

    # Set the filename
    _base_file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}"

    # Set the file info
    file_info = FileInfo(
        SubjectID=sub_id,
        TaskName=task_name,
        DatasetName="Keep Control",
        FilePath=os.path.join(_BASE_PATH, _base_file_name),
    )

    # Load the channels information
    df_channels = pd.read_csv(
        os.path.join(
            _BASE_PATH, f"sub-{sub_id}", "motion", _base_file_name + "_channels.tsv"
        ),
        sep="\t",
        header=0,
    )
    df_channels["type"] = df_channels["type"].map(_MAP_CHANNEL_TYPES)

    # Set the channel data
    channel_data = ChannelData(
        name=df_channels["name"].to_list(),
        component=df_channels["component"].to_list(),
        ch_type=df_channels["type"].to_list(),
        tracked_point=df_channels["tracked_point"].to_list(),
        units=df_channels["units"].to_list(),
        sampling_frequency=df_channels["sampling_frequency"].astype(float).to_list(),
    )

    # Load the data
    df_data = pd.read_csv(
        os.path.join(
            _BASE_PATH, f"sub-{sub_id}", "motion", _base_file_name + "_motion.tsv"
        ),
        sep="\t",
        header=0,
    )

    # Set the recording data
    recording_data = RecordingData(
        name=file_name,
        data=df_data.values,
        sampling_frequency=channel_data.sampling_frequency[0],
        times=np.arange(len(df_data)) / channel_data.sampling_frequency[0],
        channels=channel_data,
        start_time=0.0,
    )

    return MotionData(
        data=[recording_data],
        times=recording_data.times,
        info=[file_info],
        ch_names=channel_data.name,
    )
