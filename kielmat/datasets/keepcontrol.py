import numpy as np
import pandas as pd
from pathlib import Path
import openneuro
from typing import Union, Optional
from kielmat.utils.kielmat_dataclass import KielMATRecording
from kielmat.utils.kielmat_dataclass import REQUIRED_COLUMNS
import logging
import warnings

# Dict of valid tracked points for the Keep Control dataset for each tracking system
VALID_TRACKED_POINTS = {
    "omc": [
        "l_toe",
        "l_heel",
        "l_ank",
        "l_sk1",
        "l_sk2",
        "l_sk3",
        "l_sk4",
        "l_th1",
        "l_th2",
        "l_th3",
        "l_th4",
        "r_toe",
        "r_heel",
        "r_ank",
        "r_sk1",
        "r_sk2",
        "r_sk3",
        "r_sk4",
        "r_th1",
        "r_th2",
        "r_th3",
        "r_th4",
        "l_asis",
        "r_asis",
        "l_psis",
        "r_psis",
        "m_ster1",
        "m_ster2",
        "m_ster3",
        "l_sho",
        "l_ua",
        "l_elbl",
        "l_frm",
        "l_wrr",
        "l_wru",
        "l_hand",
        "r_sho",
        "r_ua",
        "r_elbl",
        "r_frm",
        "r_wrr",
        "r_wru",
        "r_hand",
        "lf_hd",
        "rf_hd",
        "lb_hd",
        "rb_hd",
        "start_1",
        "start_2",
        "end_1",
        "end_2",
    ],
    "imu": [
        "head",
        "sternum",
        "left_upper_arm",
        "left_fore_arm",
        "right_upper_arm",
        "right_fore_arm",
        "pelvis",
        "left_thigh",
        "left_shank",
        "left_foot",
        "right_thigh",
        "right_shank",
        "right_foot",
        "left_ankle",
        "right_ankle",
    ],
}


def fetch_dataset(
    dataset_path: str | Path = Path(__file__).parent / "_keepcontrol",
) -> None:
    """Fetch the Keep Control dataset from the OpenNeuro repository.
    Args:
        dataset_path (str | Path, optional): The path where the dataset is stored. Defaults to Path(__file__).parent/"_keepcontrol".
    """
    dataset_path = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path

    # Check if target folder exists, if not create it
    if not dataset_path.exists():
        dataset_path.parent.joinpath("_keepcontrol").mkdir(parents=True, exist_ok=True)

    # check if the dataset has already been downloaded (directory is not empty), if not download it
    if not any(dataset_path.iterdir()):
        # Download the dataset using openneuro-py
        openneuro.download(
            dataset="ds005258",  # this is the example Keep Control dataset on OpenNeuro, maintained by Julius Welzel
            target_dir=dataset_path,
        )

    return


def load_recording(
    dataset_path: str | Path = Path(__file__).parent / "_keepcontrol",
    id: str = "pp001",
    task: str = "walkSlow",
    tracking_systems: Union[str, list[str]] = ["imu", "omc"],
    tracked_points: Optional[Union[None, str, list[str]]] = None,
):
    """
    Load a recording from the Keep Control validation study.
    Args:
        dataset_path (str or Path, optional): The path to the dataset. Defaults to the "_keepcontrol" directory in the same directory as this file.
        id (str): The ID of the recording.
        tracking_systems (str or list of str): A string or list of strings representing the tracking systems for which data should be returned.
        tracked_points (None, str or list of str, optional): The tracked points of interest. If None, all tracked points will be returned. Defaults to None.
    Returns:
        KielMATRecording: An instance of the KielMATRecording dataclass containing the loaded data and channels.
    """

    # Fetch the dataset if it does not exist
    if not dataset_path.exists():
        fetch_dataset()

    # check if id contains sub or sub- substring, if so replace it with ''
    id = id.replace("sub", "").replace("-", "")

    # check if task contains task or task- substring, if so replace it with ''
    task = task.replace("task", "").replace("-", "")

    # Put tracking systems in a list
    if isinstance(tracking_systems, str):
        tracking_systems = [tracking_systems]

    # check if tracked points has been specified, if not use all tracked points
    if tracked_points is None:
        tracked_points = {
            tracksys: VALID_TRACKED_POINTS[tracksys] for tracksys in tracking_systems
        }
    # Tracked points will be a dictionary mapping
    # each tracking system to a list of tracked points of interest
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]
    if isinstance(tracked_points, list):
        tracked_points = {tracksys: tracked_points for tracksys in tracking_systems}
    # use the VALID_TRACKED_POINTS dictionary to get the valid tracked points for each tracking system
    # return error of some tracked_points are not valid
    # log which of the specified tracked points are not valid
    for tracksys in tracking_systems:
        if not all(
            tracked_point in VALID_TRACKED_POINTS[tracksys]
            for tracked_point in tracked_points[tracksys]
        ):
            logging.warning(f"Invalid tracked points for tracking system {tracksys}.")
            logging.warning(
                f"Valid tracked points are: {VALID_TRACKED_POINTS[tracksys]}"
            )
            invalid_points = [
                tracked_point
                for tracked_point in tracked_points[tracksys]
                if tracked_point not in VALID_TRACKED_POINTS[tracksys]
            ]
            logging.warning(f"Invalid tracked points are: {invalid_points}")
            return

    # Load data and channels for each tracking system
    data_dict, channels_dict = {}, {}
    for tracksys in tracking_systems:
        # Find avaliable files for the give ID and task and tracking system
        file_name = list(
            dataset_path.glob(
                f"sub-{id}/motion/sub-{id}_task-{task}_tracksys-{tracksys}_*motion.tsv"
            )
        )
        # check if file exists, if not log message and return
        if not file_name:
            logging.warning(
                f"No files found for ID {id}, task {task}, and tracking system {tracksys}."
            )
            return
        # check if multiple files are found, if so log message and return
        if len(file_name) > 1:
            logging.warning(
                f"Multiple files found for ID {id}, task {task}, and tracking system {tracksys}."
            )
            return

        # Load the data and channels for the tracking system
        df_data = pd.read_csv(file_name[0], sep="\t")
        df_channels = pd.read_csv(
            file_name[0].parent
            / f"sub-{id}_task-{task}_tracksys-{tracksys}_channels.tsv",
            sep="\t",
        )

        # filter the data and channels to only include the tracked points of interest
        df_channels = df_channels[
            df_channels["tracked_point"].isin(tracked_points[tracksys])
        ]

        # only keep df_data columns that are in df_channels
        col_names = df_channels["name"].values
        df_data = df_data[col_names]

        # transform the data and channels into a dictionary for the KielMATRecording dataclass
        data_dict[tracksys] = df_data
        channels_dict[tracksys] = df_channels

    # construct data class
    recording = KielMATRecording(data=data_dict, channels=channels_dict)

    # add information about the recording to the data class
    recording.add_info("Subject", id)
    recording.add_info("Task", task)

    return recording
