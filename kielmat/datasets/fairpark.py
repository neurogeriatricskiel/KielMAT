import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import pathlib
import polars as pl
from kielmat.utils.kielmat_dataclass import KielMATRecording


SAMPLING_FREQ_HZ: float = 100.0  # sampling frequency

MAP_CHANNEL_UNITS = {"ACCEL": "m/s^2", "ANGVEL": "deg/s", "MAGN": "Gauss"}


def load_recording(
    file_name: str | pathlib.Path,
    tracking_systems: str | list[str] = ["imu"],
    tracked_points: str | list[str] | dict[str, str] | dict[str, list[str]] = {
        "imu": ["LARM"]
    },
) -> KielMATRecording:
    """
    Load a recording from the FAIRPARK II validation study.

    Args:
        file_name (str or pathlib.Path ): The absolute or relative path to the data file.
        tracking_systems (str or list of str) : A string or list of strings of tracking systems for which data are to be returned.
        tracked_points (str or list of str or dict[str, str] or dict[str, list of str]) :
            Defines for which tracked points data are to be returned.
            If a string or list of strings is provided, then these will be mapped to each requested tracking system.
            If a dictionary is provided, it should map each tracking system to either a single tracked point or a list of tracked points.

    Returns:
        KielMATRecording : An instance of the KielMATRecording dataclass containing the loaded data and channels.
    """
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

    # Extract relevant metadata from filename
    file_name_parts = os.path.splitext(os.path.basename(file_name))[0].split("_")
    tracked_point = next((part for part in file_name_parts if "imu-" in part), None)
    if tracked_point:
        tracked_point = tracked_point.split("-")[-1]

    # Load the data from the file
    col_names = [
        f"{tracked_point}_{s}_{x}"
        for s in ["ACCEL", "ANGVEL", "MAGN"]
        for x in ["x", "y", "z"]
    ] + ["year", "month", "day", "hour", "minute", "second"]
    df = pd.read_csv(file_name, header=None, sep=";", names=col_names)
    timestamps = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute", "second"]]
    )
    df["timestamp"] = pd.date_range(
        start=timestamps.iloc[0],
        periods=len(timestamps),
        freq=f"{int(1/SAMPLING_FREQ_HZ*1000)}ms",
    )
    df.set_index(df["timestamp"], inplace=True, drop=True)
    df.drop(
        labels=["year", "month", "day", "hour", "minute", "second", "timestamp"],
        axis=1,
        inplace=True,
    )

    # Make the channel dictionary
    channels_dict = {
        "name": df.columns.to_list(),
        "component": [f"{x}" for _ in range(3) for x in ["x", "y", "z"]],
        "type": [f"{s}" for s in ["ACCEL", "ANGVEL", "MAGN"] for _ in range(3)],
        "tracked_point": [tracked_point for _ in range(len(df.columns))],
        "units": [
            f"{MAP_CHANNEL_UNITS[s]}"
            for s in ["ACCEL", "ANGVEL", "MAGN"]
            for _ in range(3)
        ],
        "sampling_frequency": [SAMPLING_FREQ_HZ for _ in range(len(df.columns))],
    }

    return KielMATRecording(
        data={"imu": df}, channels={"imu": pd.DataFrame(channels_dict)}
    )
