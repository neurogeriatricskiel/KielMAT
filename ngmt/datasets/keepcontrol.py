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

_MAP_CHANNEL_TYPES = {
    "ACC": "ACCEL",
    "ANGVEL": "GYRO",
    "MAGN": "MAGN"
}

if sys.platform == "linux":
    _BASE_PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
elif sys.platform == "win32":
    _BASE_PATH = "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
else:
    raise OSError(
        "Currently only Windows- or Linux-based operating systems are supported!"
    )


<<<<<<< HEAD
def load_file(sub_id: str, task_name: str, tracksys: str) -> MotionData:
=======
def load_file(sub_id: str, task_name: str, tracksys: str) -> IMUDataset:
    """
    Args:
        sub_id (str): _description_
        task_name (str): _description_
        tracksys (str): _description_

    Returns:
        IMUDataset: _description_
    """
>>>>>>> b95f2456eb5ff52b4964a16bb29ff8b2d7f92f37
    # Set the filename
    _base_file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}"

    # Set the file info
    file_info = FileInfo(
        SubjectID=sub_id,
        TaskName=task_name,
        DatasetName="Keep Control",
        FilePath=os.path.join(_BASE_PATH, _base_file_name)
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
        sampling_frequency=df_channels["sampling_frequency"].astype(float).to_list()
    )

    # Load the data
    df_data = pd.read_csv(
        os.path.join(
            _BASE_PATH, f"sub-{sub_id}", "motion", _base_file_name + "_motion.tsv"
        ),
        sep="\t",
        header=0,
    )

    # For each tracked point,
    imus = []
    for tracked_point in df_channels["tracked_point"].unique():
        imus.append(
            IMUDevice(
                tracked_point=tracked_point,
                recordings=[
                    IMURecording(
                        type=sensor_type,
                        units=df_channels[
                            (df_channels["tracked_point"] == tracked_point)
                            & (df_channels["type"] == sensor_type)
                        ]["units"].iloc[0],
                        fs=df_channels[
                            (df_channels["tracked_point"] == tracked_point)
                            & (df_channels["type"] == sensor_type)
                        ]["sampling_frequency"]
                        .iloc[0]
                        .astype(float),
                        data=df_data.loc[
                            :,
                            [
                                col
                                for col in df_data.columns
                                if (tracked_point in col) and (sensor_type in col)
                            ],
                        ].values,
                    )
                    for sensor_type in df_channels[
                        (df_channels["tracked_point"] == tracked_point)
                    ]["type"].unique()
                ],
            )
        )

    return IMUDataset(subject_id=sub_id, devices=imus)
