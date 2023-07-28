import numpy as np
import pandas as pd
from ngmt.motiondata import MotionData, FileInfo, ChannelMetaData
from datetime import datetime, timedelta

def import_polar_watch(data_file_path):
    raw = pd.read_csv(data_file_path)

    # Extract the basic information from the first row (athlete data)
    athlete_data = raw.iloc[0]
    athlete_info = FileInfo(
        TaskName=athlete_data['Name'],
        SamplingFrequency=np.nan,  # Not available in the athlete data
        TaskDescription=f"{athlete_data['Sport']} on {athlete_data['Date']} at {athlete_data['Start time']}",
    )

    # Drop the athlete data and remaining rows with NaN values
    df = raw.iloc[2:]


    # Create timestamps for the motion data
    start_time = pd.to_datetime(f"{athlete_data['Date']} {athlete_data['Start time']}")
    times_str = df['Sport'].to_list()
    times = [(datetime.strptime(time_str, '%H:%M:%S').hour * 3600 +
                 datetime.strptime(time_str, '%H:%M:%S').minute * 60 +
                 datetime.strptime(time_str, '%H:%M:%S').second) for time_str in times_str]


    df = pd.read_csv(data_file_path, skiprows=[0, 1])  # Skip the first and second rows

    # Extract channel names
    channel_names = ['heart_rate', 'walking_velocity', 'position_from_start']

    # Extract the time series data
    time_series = df[['HR (bpm)', 'Speed (km/h)', 'Distances (m)']].values.T

    # Create ChannelMetaData objects for each channel
    channels = []
    for channel_name in channel_names:
        channel_data = ChannelMetaData(name=channel_name, component="n/a", ch_type="MISC", tracked_point="n/a", units="")
        channels.append(channel_data)

    # Create the MotionData object
    motion_data = MotionData(
        info=athlete_info,
        times=times,
        channel_names=channel_names,
        time_series=time_series,
    )

    return motion_data