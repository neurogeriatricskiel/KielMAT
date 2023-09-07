from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

VALID_CHANNEL_TYPES = {
    "ACCEL",
    "ANGACCEL",
    "GYRO",
    "JNTANG",
    "LATENCY",
    "MAGN",
    "MISC",
    "ORNT",
    "POS",
    "VEL",
}
VALID_COMPONENT_TYPES = {"x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "n/a"}


@dataclass
class FileInfo:
    """
    A data class representing information about a file to be processed.

    Attributes:
        SubjectID (str): The identifier of the subject associated with the file.
        TaskName (str): The name of the task or experiment associated with the file.
        DatasetName (str): The name of the dataset to which the file belongs.
        FilePath (str): The path to the file in the file system.
    """

    SubjectID: str
    TaskName: str
    DatasetName: str
    FilePath: str


@dataclass
class ChannelData:
    """
    A data class for representing information about channels used in a recording.

    Attributes:
        name (List[str]): A list of channel names.
        component (List[str]): A list of channel components.
        ch_type (List[str]): A list of channel types.
        tracked_point (List[str]): A list of tracked points.
        units (List[str]): A list of measurement units.
        placement (Optional[List[str]]): An optional list of placement information (default is None).
        description (Optional[List[str]]): An optional list of channel descriptions (default is None).
        sampling_frequency (Optional[float]): An optional sampling frequency (default is None).
        status (Optional[List[float]]): An optional list of channel statuses (default is None).
        status_description (Optional[List[str]]): An optional list of status descriptions (default is None).

    Raises:
        ValueError: If the provided 'ch_type' is not one of the valid channel types.
        ValueError: If the provided 'component' is not one of the valid component types.
    """

    name: List[str]
    component: List[str]
    ch_type: List[str]
    tracked_point: List[str]
    units: List[str]
    placement: Optional[List[str]] = None
    description: Optional[List[str]] = None
    sampling_frequency: Optional[float] = None
    status: Optional[List[float]] = None
    status_description: Optional[List[str]] = None

    def __post_init__(self):
        if self.ch_type not in VALID_CHANNEL_TYPES:
            raise ValueError(
                f"Invalid channel type {self.ch_type}. Must be one of {VALID_CHANNEL_TYPES}"
            )
        if self.component not in VALID_COMPONENT_TYPES:
            raise ValueError(
                f"Invalid component type {self.component}. Must be one of {VALID_COMPONENT_TYPES}"
            )


@dataclass
class EventData:
    """
    A data class to describe timing and other properties of events during a recording.
    Events can include stimuli presented to the participant,
    participant responses or labeling of data samples.
    Events can overlap in time.

    Attributes:
        name (List[str]): A list of event names.
        onset (List[float]): A list of event onset times.
        duration (List[float]): A list of event durations.
        trial_type (Optional[List[str]]): An optional list of trial types (default is None).
    """

    name: List[str]
    onset: List[float]
    duration: List[float]
    trial_type: Optional[List[str]] = None


@dataclass
class RecordingData:
    """
    A data class to hold meaningful groups of motion data channels from a single
    recording. For example, a recording of a participant walking on a treadmill.

    Attributes:

        name (str): A name for the recording data.

        data (np.ndarray): A nD numpy array of shape (n_channels, n_samples) containing
            the motion data. Channels MUST have the same sampling frequency.

        sampling_frequency (float): The sampling frequency of the motion data.

        times (np.ndarray): A 1D numpy array of shape (n_samples,) containing the
            timestamps of the motion data. If no time stamps are provided from the
            system, timestamps are relative to the start of the recording.

        channels (ChannelData): A ChannelData object containing information about the
            channels in the data.

        start_time (float): The start time of the recording in seconds. 0 if no time
            stamps are provided from the system.

        types (List[str]): A list of strings describing the type of data in each channel.
            For example, "acceleration", "angular velocity", "position", etc.

        ch_names (Optional[List[str]]): An optional list of channel names (default is None).
            If None, the channel names will be set to the channel numbers.

        events (Optional[EventData]): An optional EventData object containing information
            about events during the recording (default is None).
    """

    name: str
    data: np.ndarray
    sampling_frequency: float
    times: np.ndarray
    channels: ChannelData
    start_time: float
    types: List[str]
    ch_names: Optional[List[str]] = None
    events: Optional[EventData] = None


@dataclass
class MotionData:
    data: List[RecordingData]
    times: np.ndarray  # Can be a 1D array representing timestamps
    info: List[FileInfo]
    ch_names: List[str]  # Can be a list of channel names

    def __post_init__(self):
        if len(self.times) != self.time_series.shape[1]:
            raise ValueError(
                "The length of `times` should match the number of columns in `time_series`"
            )

        if len(self.channel_names) != self.time_series.shape[0]:
            raise ValueError(
                "The number of `channel_names` should match the number of rows in `time_series`"
            )

    @classmethod
    def synchronise_recordings(self, systems: List[RecordingData]):
        """This functions uses the times provided from the systems to create a globally
        valid times vector. If no start time is provided or if the start times start at
        0, meaningful synchronization is not possible functions returns a warning.


        Args:
            systems (List[RecordingData]): List of RecordingData objects

        """

        # find the start time of each system if not 0
        start_times = []
        for system in systems:
            if system.start_time != 0:
                start_times.append(system.start_time)

        # if all start times are 0, no meaningful synchronization is possible, give warning
        if len(start_times) == 0:
            print(
                "Warning: No start times provided or all start times are 0. No meaningful synchronization is possible."
            )
            return

        # find the minimum start time and the index of the system with the minimum start time
        min_start_time = min(start_times)
        min_start_time_index = start_times.index(min_start_time)

        # find the time difference between the start time of the system with the minimum start time and the other systems
        time_diffs = []
        for system in systems:
            time_diffs.append(system.start_time - min_start_time)

        # find the highest time stamp in regard to the system with the minimum start time
        max_time = 0
        for system in systems:
            if system.times[-1] > max_time:
                max_time = system.times[-1]

        # find the system with the highest sampling frequency
        max_sampling_frequency = 0
        for system in systems:
            if system.sampling_frequency > max_sampling_frequency:
                max_sampling_frequency = system.sampling_frequency

        # create a new times vector with the highest sampling frequency and the highest time stamp
        new_times = np.arange(0, max_time, 1 / max_sampling_frequency)

        # store new time vector
        self.times = new_times

        return
