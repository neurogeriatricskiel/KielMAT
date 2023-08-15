from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from ngmt.file_io import get_unit_from_type

@dataclass
class FileInfo:
    SubjectId: str
    TaskName: str
    SamplingFrequency: float
    TaskDescription: Optional[str] = None
    Instructions: Optional[str] = None
    Manufacturer: Optional[str] = None
    ManufacturersModelName: Optional[str] = None
    MissingValues: Optional[str] = None
    SamplingFrequencyEffective: Optional[float] = None
    TrackedPointsCount: Optional[int] = None
    TrackingSystemName: Optional[str] = None


VALID_CHANNEL_TYPES = {"ACCEL", "ANGACCEL", "GYRO", "JNTANG", "LATENCY", "MAGN", "MISC", "ORNT", "POS", "VEL"}
VALID_COMPONENT_TYPES = {"x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "n/a"}

@dataclass
class ChannelMetaData:
    name: list[int] = field(default_factory=list)
    component: list[str] = field(default_factory=list)
    ch_type: list[str] = field(default_factory=list)
    tracked_point: list[int] = field(default_factory=list)
    units: list[int] = field(default_factory=list)
    placement: Optional[str] = None
    description: Optional[str] = None
    sampling_frequency: Optional[float] = None
    status: Optional[str] = None
    status_description: Optional[str] = None
    additional_columns: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        if not all(ch in VALID_CHANNEL_TYPES for ch in self.ch_type):
            raise ValueError(f"Invalid channel type {self.ch_type}. Must be one of {VALID_CHANNEL_TYPES}")
        if not all(comp in VALID_COMPONENT_TYPES for comp in self.component):
            raise ValueError(f"Invalid component type {self.component}. Must be one of {VALID_COMPONENT_TYPES}")

@dataclass
class MotionData:
    info: FileInfo  # Can be a FileInfo object
    channels: ChannelMetaData  # Can be a ChannelMetaData object
    times: np.ndarray  # Can be a 1D array representing timestamps
    time_series: np.ndarray  # Can be a 2D array where each row is a channel

    def __post_init__(self):
        if len(self.times) != self.time_series.shape[1]:
            raise ValueError("The length of `times` should match the number of columns in `time_series`")

        if len(self.channels.name) != self.time_series.shape[0]:
            raise ValueError("The number of `channel_names` should match the number of rows in `time_series`")

    @classmethod
    def import_hasomed_imu(self, file: str):
        """
        This function reads a file and returns a MotionData object. 

        Parameters:
        file (str): path to the .csv file

        Returns:
        MotionData: an object of class MotionData that includes FileInfo 
        object with metadata from the file,
        a 1D numpy array with time values, 
        a list of channel names, 
        and a 2D numpy array with the time series data.

        The file structure is assumed to be as follows:
        - The header contains lines starting with '#' with metadata information.
        - The actual time series data starts after the metadata lines.
        - The first column in the time series data represents the 'Sample number' or time.
        - The remaining columns represent the channel data.

        Note: This function only extracts a subset of the possible FileInfo fields.
        Additional fields need to be added manually depending on what fields are present in the files.
        Also, error checking and exception handling has been kept minimal for simplicity.
        You might want to add more robust error handling for a production-level application.
        """
        with open(file, 'r') as f:
            lines = f.readlines()

        # Keep track of where the metadata ends and the time series data begins
        data_start_idx = 0

        # Instantiate empty FileInfo
        info = FileInfo(SubjectId="", TaskName="", SamplingFrequency=100.0)  # default SamplingFrequency to 100.0 if not found

        for idx, line in enumerate(lines):
            # Metadata ends when we encounter a line that doesn't start with '#'
            if not line.startswith('#'):
                data_start_idx = idx
                break

            # Extract fields for FileInfo
            parts = line.strip().split(';')

            if "Patient-ID" in line:
                info.SubjectId = parts[2]
            elif "Assessment type" in line:
                info.TaskName = parts[3]
            elif "Sample rate" in line:
                info.SamplingFrequency = float(parts[6])

            # Add more fields as necessary here...

        # Create DataFrame from the time series data
        data = pd.read_csv(file, skiprows=data_start_idx - 1, delimiter=';')

        # Extract the channel names from the column names of the DataFrame
        channel_names = data.columns.tolist()

        # Convert time to numpy array
        times = np.linspace(0, data.shape[0] / info.SamplingFrequency, data.shape[0])

        type_imu = ['Acc', 'Gyro', 'Mag'] 
        # drop non relevant columns
        filtered_col_names = [col for col in channel_names if not any(sensor in col for sensor in type_imu)]
        channel_names = [col for col in channel_names if any(sensor in col for sensor in type_imu)]
        # have consited numbering in channel_names and replace . with _
        channel_names = [col.replace('.', '_') for col in channel_names]
        # check if channel_name is numbered, if not append 0
        for idx, name in enumerate(channel_names):
            if not name[-1].isdigit():
                channel_names[idx] = name + '_0'

        # define ChannelMetaData
        # extract channel component
        channel_components = [name.split('_')[-2][-1].lower() for name in channel_names]
        # extract channel type
        channel_types = [name.split('_')[0][:-1] for name in channel_names]
        # convert to meaningful types (Acc -> ACCEL, Gyro -> GYRO, Mag -> MAGN)
        channel_types = ['ACCEL' if 'Acc' in type else 'GYRO' if 'Gyro' in type else 'MAGN' for type in channel_types]
        # get channel units
        channel_units = get_unit_from_type(channel_types)
        # get tracked point
        channels_tracked_point = [name.split('_')[0][-1] for name in channel_names]

        # update ChannelMetaData
        channels = ChannelMetaData(name=channel_names,
                                        component=channel_components,
                                        ch_type=channel_types,
                                        tracked_point=channels_tracked_point,
                                        units=channel_units)

        time_series = data.drop(columns=filtered_col_names).to_numpy().T  # transpose

        return self(info=info, channels=channels, times=times, time_series=time_series)
