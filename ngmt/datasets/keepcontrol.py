import pandas as pd
import os
import sys
from ..utils.data_utils import IMUDataset, IMUDevice, IMURecording

if sys.platform == "linux":
    _BASE_PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
elif sys.platform == "win32":
    _BASE_PATH = "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
else:
    raise OSError(
        "Currently only Windows- or Linux-based operating systems are supported!"
    )


def load_file(sub_id: str, task_name: str, tracksys: str) -> IMUDataset:
    # Set the filename
    _base_file_name = f"sub-{sub_id}_task-{task_name}_tracksys-{tracksys}"

    # Load the channels information
    df_channels = pd.read_csv(
        os.path.join(
            _BASE_PATH, f"sub-{sub_id}", "motion", _base_file_name + "_channels.tsv"
        ),
        sep="\t",
        header=0,
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
