import numpy as np
import pandas as pd
from pathlib import Path
import openneuro
import os
from ngmt.utils.ngmt_dataclass import NGMTRecording
from ngmt.utils.ngmt_dataclass import REQUIRED_COLUMNS

def fetch_dataset(
    dataset_path: str | Path = Path(__file__).parent / "_keepcontrol",
) -> None:
    """Fetch the Keep Control dataset from the OpenNeuro repository.

    Args:
        progressbar (bool, optional): Whether to display a progressbar. Defaults to True.
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
            dataset="ds005258", # this is the example Keep Control dataset on OpenNeuro, maintained by Julius Welzel
            target_dir=dataset_path,
        )
        
        print(f"Dataset has been downloaded to {dataset_path}")

    # print message to user if dataset has already been downloaded
    else:
        print(f"Dataset has already been downloaded to {dataset_path}")

    return
        
def load_recording(
    file_name: Path,
    tracking_systems: str | list[str],
    tracked_points: str | list[str] | dict[str, str] | dict[str, list[str]],
):
    """
    Load a recording from the Keep Control validation study.

    Args:
        file_name (pathlib.Path): The absolute or relative path to the data file.
        tracking_systems (str or list of str) : A string or list of strings of tracking systems for which data are to be returned.
        tracked_points (str or list of str or dict[str, str] or dict[str, list of str]) :
            Defines for which tracked points data are to be returned.
            If a string or list of strings is provided, then these will be mapped to each requested tracking system.
            If a dictionary is provided, it should map each tracking system to either a single tracked point or a list of tracked points.

    Returns:
        NGMTRecording : An instance of the NGMTRecording dataclass containing the loaded data and channels.
    """

    # Fetch the dataset if it does not exist
    progressbar = False if not progressbar else progressbar
    file_path = Path(dataset_path) / cohort / file_name
    if not file_path.exists():
        fetch_dataset()

    # Put tracking systems in a list
    if isinstance(tracking_systems, str):
        tracking_systems = [tracking_systems]

    # Tracked points will be a dictionary mapping
    # each tracking system to a list of tracked points of interest
    if isinstance(tracked_points, str):
        tracked_points = [tracked_points]
    if isinstance(tracked_points, list):
        tracked_points = {tracksys: tracked_points for tracksys in tracking_systems}
    for k, v in tracked_points.items():
        if isinstance(v, str):
            tracked_points[k] = [v]

    # From the file_name, extract the tracking system
    search_str = "_tracksys-"
    idx_from = str(file_name).find(search_str) + len(search_str)
    idx_to = idx_from + str(file_name)[idx_from:].find("_")
    current_tracksys = str(file_name)[idx_from:idx_to]

    # Initialize the data and channels dictionaroes
    data_dict, channels_dict = {}, {}
    for tracksys in tracking_systems:
        # Set current filename
        current_file_name = str(file_name).replace(
            f"{search_str}{current_tracksys}", f"{search_str}{tracksys}"
        )
        if os.path.isfile(current_file_name):
            # Read the data and channels info into a pandas DataFrame
            df_data = pd.read_csv(current_file_name, header=0, sep="\t")
            df_channels = pd.read_csv(
                current_file_name.replace("_motion.tsv", "_channels.tsv"),
                header=0,
                sep="\t",
            )

            # Now select only for the tracked points of interest
            df_data = df_data.loc[
                :,
                [
                    col
                    for col in df_data.columns
                    if any(
                        [
                            tracked_point in col
                            for tracked_point in tracked_points[tracksys]
                        ]
                    )
                ],
            ]
            if df_data.empty: 
                print(f"Specified tracked point not found in the tracking system {tracksys}.")
            
            df_channels = df_channels[
                df_channels["tracked_point"].str.contains("|".join(tracked_points[tracksys]))
            ]

            # Add sampling frequency to the channels DataFrame
            df_channels["sampling_frequency"] = 200

            # Put data and channels in output dictionaries
            col_names = [c for c in REQUIRED_COLUMNS] + [
                c for c in df_channels.columns if c not in REQUIRED_COLUMNS
            ]
            data_dict[tracksys] = df_data
            channels_dict[tracksys] = df_channels[col_names]
    return NGMTRecording(data=data_dict, channels=channels_dict)
