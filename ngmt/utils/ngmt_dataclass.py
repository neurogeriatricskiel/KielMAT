from bids_validator import BIDSValidator
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Optional, Union, Sequence

# from ngmt.modules import GSDB

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type
VALID_CHANNEL_TYPES = {
    "ACCEL",
    "ANGACCEL",
    "BARO",
    "GYRO",
    "JNTANG",
    "LATENCY",
    "MAGN",
    "MISC",
    "ORNT",
    "POS",
    "VEL",
}

# See: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-component
VALID_COMPONENT_TYPES = {"x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "n/a"}

# See https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html#participants-file
VALID_INFO_KEYS = {
    "Subject",
    "Session",
    "Task",
}

        
VALID_CHANNEL_STATUS_VALUES = ["good", "bad", "n/a"]


@dataclass(kw_only=True)
class NGMTRecording:
    """Dataclass to hold any data and associated infos for a NGMT recording.

    Attributes:
        data (dict): The data is stored as a pandas DataFrame for each unique tracking system.
        channels (dict): The channels descriptions are stored as a pandas DataFrame for each unique tracking system.
        info (dict): The infos on the subject, task, and more, are stored as a nested dictionary.
        events (dict): The events are stored as a pandas DataFrame for each unique tracking system.
        events_info (dict): The event infos are stored as a nested dictionary.
    """

    data: dict[str, pd.DataFrame]
    channels: dict[str, pd.DataFrame]
    info: None | dict[str, Any] = None
    events: None | dict[str, pd.DataFrame] = None
    events_info: None | dict[str, Any] = None

    def __post_init__(self):
        # Validate channels when an instance is created
        self.validate_channels()

    def validate_channels(self):
        """
        Validates the channel dataframes for each system.

        This function checks if the channel dataframes have the required columns in the correct order,
        and if the data types of the columns are valid. It also performs additional value checks for
        optional columns.

        Raises:
            ValueError: If the channel dataframe does not have the required columns in the correct order,
                or if the 'component' column contains invalid values, or if the 'type' column is not
                uppercase strings, or if the 'status' column contains invalid values.
            TypeError: If the 'name' column is not of type string.

        Returns:
            str: A message indicating that all channel dataframes are valid.
        """
        required_columns = ['name', 'component', 'type', 'tracked_point', 'units', 'sampling_frequency']

        for system_name, df in self.channels.items():
            # Check required columns and their order
            if not df.columns.tolist()[:6] == required_columns:
                raise ValueError(f"Channel dataframe for '{system_name}' does not have the required columns in correct order. The correct order is: {required_columns}")

            # Check data types
            if not all(isinstance(name, str) for name in df['name']):
                raise TypeError(f"Column 'name' in '{system_name}' must be of type string.")
            if not all(item in VALID_COMPONENT_TYPES for item in df['component']):
                raise ValueError(f"Column 'component' in '{system_name}' contains invalid values.")
            if not all(isinstance(typ, str) and typ.isupper() for typ in df['type']):
                raise ValueError(f"Column 'type' in '{system_name}' must be uppercase strings.")

            # Additional value checks for optional columns
            if 'status' in df.columns and not all(s in VALID_CHANNEL_STATUS_VALUES for s in df['status'] if s != 'n/a'):
                raise ValueError(f"Column 'status' in '{system_name}' contains invalid values.")

        return "All channel dataframes are valid."

    def add_events(self, tracking_system: str, new_events: pd.DataFrame) -> None:
        """Add events to the recording for a specific tracking system.

        Args:
            tracking_system (str): Tracking system for which events are to be added.
            new_events (pd.DataFrame): Events to be added in BIDS format.
        """
        if self.events is None:
            self.events = {}

        if tracking_system not in self.events:
            self.events[tracking_system] = new_events
        else:
            existing_events = self.events[tracking_system]
            self.events[tracking_system] = pd.concat(
                [existing_events, new_events], ignore_index=True
            )

    def add_info(self, key: str, value: Any) -> None:
        """Add information to the info dictionary. Valid keys are : 'Subject', 'Session', 'Task'.

        Args:
            key (str): The key for the information.
            value (Any): The value of the information.

        Raises:
            ValueError: If the provided 'key' is not one of the valid info keys.

        Examples:
            >>> recording.add_info("Subject", "01")
        """
        if self.info is None:
            self.info = {}

        # Check if the key belongs to a list of keywords
        if key not in VALID_INFO_KEYS:
            print(
            f"Warning: Invalid info key '{key}'. Valid info keys are: {VALID_INFO_KEYS}"
            )

        # add the key-value pair to the info dictionary
        self.info[key] = value

        # Check if the value are lower case, if not, convert to lower case and give warning
        if isinstance(value, str):
            self.info[key] = value.lower()
            print(
                f"Warning: The value of the key '{key}' should be lower case. Converted to lower case."
            )

        # check if value contains underscore or space, if yes, remove and give warning
        if "_" in value or " " in value:
            self.info[key] = value.replace("_", "").replace(" ", "")
            print(
                f"Warning: The value of the key '{key}' should not contain underscore or space. Removed underscore and space."
            )

    def export_events(
        self,
        file_path: str,
        tracking_system: Optional[str] = None,
        file_name: Optional[str] = None,
        bids_compatible_fname: Optional[bool] = False,
    ) -> None:
        """Export events for a specific tracking system to a file.

        Args:
            tracking_system (Optional[str]): Tracking system for which events are to be exported.
                If None, events from all tracking systems will be exported (default is None).
            file_path (str): Path to the directory where the file should be saved.
            file_name (Optional[str]): Name of the file to be exported. If None, a default name will be used.
            bids_compatible_fname (bool): Flag indicating whether the exported filename should be BIDS compatible (default is False).
        """
        if self.events is not None:
            if tracking_system is None:
                all_events = pd.concat(
                    self.events.values(),
                    keys=self.events.keys(),
                    names=["tracking_system"],
                )
                if file_name is None:
                    file_name = "all_events.csv"
                if bids_compatible_fname:
                    # Construct the filename using subject ID and task name
                    subject_id = self.info.get("Subject", "")
                    task_name = self.info.get("Task", "")
                    # check if subject_id and task_name are present in the info dictionary
                    if subject_id == None or task_name == None:
                        raise ValueError(
                            "Subject ID and Task Name should be specified in the info dictionary."
                        )
                    file_name = f"sub-{subject_id}_task-{task_name}_events.csv"
                    # check if session is present in the info dictionary
                    session = self.info.get("Session")
                    if session != None:
                        file_name = f"sub-{subject_id}_ses-{session}_task-{task_name}_events.csv"
                    file_path = Path(file_path).joinpath(file_name)
                    all_events.to_csv(file_path, sep="\t", index=False)
                else:
                    file_path = Path(file_path).joinpath(file_name)
                    all_events.to_csv(file_path, index=False)
            elif tracking_system in self.events:
                if file_name is None:
                    file_name = f"{tracking_system}_events.csv"
                if bids_compatible_fname:
                    file_name = file_name.replace(".csv", "_events.tsv")
                    file_path = Path(file_path).joinpath(file_name)
                    self.events[tracking_system].to_csv(
                        file_path, sep="\t", index=False
                    )
                else:
                    file_path = Path(file_path).joinpath(file_name)
                    self.events[tracking_system].to_csv(file_path, index=False)

            # check if file_path is BIDS compatible
            if bids_compatible_fname:
                # validate the file_path
                validator = BIDSValidator()
                errors = validator.is_bids(file_path)
                if errors:
                    raise ValueError(f"File path '{file_path}' is not BIDS compatible.")


# @dataclass
# class RecordingData:
#     """
#     A data class to hold meaningful groups of motion data channels from a single
#     recording. For example, a recording of a participant walking on a treadmill.

#     Attributes:
#         name (str): A name for the recording data.
#         data (np.ndarray): A nD numpy array of shape (n_samples, n_channels) containing
#             the motion data. Channels MUST have the same sampling frequency.
#         sampling_frequency (float): The sampling frequency of the motion data.
#         times (np.ndarray): A 1D numpy array of shape (n_samples,) containing the
#             timestamps of the motion data. If no time stamps are provided from the
#             system, timestamps are relative to the start of the recording.
#         channels (ChannelData): A ChannelData object containing information about the
#             channels in the data.
#         start_time (float): The start time of the recording in seconds. 0 if no time
#             stamps are provided from the system.
#         events [EventData]: An  EventData object containing information
#             about events during the recording (default is None for all inputs).
#     """

#     name: str
#     data: np.ndarray
#     sampling_frequency: float
#     times: np.ndarray
#     channels: ChannelData
#     start_time: float
#     events: EventData = None

#     def __post_init__(self):
#         if len(self.times) != self.data.shape[0]:
#             raise ValueError(
#                 "The length of `times` should match the number of rows in `data`"
#             )

#     def pick_channel_types(self, channel_type_oi):
#         """
#         This function returns a trimmed version of the RecordingData
#         for a given channel type.

#         Parameters:
#             channel_type_oi (str): channel type

#         Returns:
#             recording_data_clean_type (RecordingData): An object of class RecordingData that includes
#                 FileInfo object with metadata from the file, a 1D numpy array
#                 with time values, a list of channel names, and a 2D numpy array
#                 with the time series data.
#         """

#         # find the indices_type_oi of the channels with the given channel type
#         indices_type_oi = []
#         for index, channel_type in enumerate(self.channels.ch_type):
#             if channel_type == channel_type_oi:
#                 indices_type_oi.append(index)

#         # iterate through the indices_type_oi and create a new ChannelData object
#         channel_data_clean_type = ChannelData(
#             name=[self.channels.name[index] for index in indices_type_oi],
#             component=[self.channels.component[index] for index in indices_type_oi],
#             ch_type=[self.channels.ch_type[index] for index in indices_type_oi],
#             tracked_point=[
#                 self.channels.tracked_point[index] for index in indices_type_oi
#             ],
#             units=[self.channels.units[index] for index in indices_type_oi],
#         )
#         # create a new MotionData object with the given channel type
#         recording_data_clean_type = RecordingData(
#             name=self.name,
#             data=self.data[:, indices_type_oi],
#             sampling_frequency=self.sampling_frequency,
#             times=self.times,
#             channels=channel_data_clean_type,
#             start_time=self.start_time,
#             events=self.events,
#         )

#         return recording_data_clean_type

#     def pick_channels(self, channel_names_oi):
#         """
#         This function returns a trimmed version of the RecordingData
#         for a given channel type.

#         Parameters:
#             channel_names_oi (str): channel names or unique substring in channel names of interest

#         Returns:
#             recording_data_clean_type (RecordingData): An object of class RecordingData that includes
#                 FileInfo object with metadata from the file, a 1D numpy array
#                 with time values, a list of channel names, and a 2D numpy array
#                 with the time series data.
#         """

#         # find the indices_type_oi of the channels with the given channel type
#         indices_type_oi = []
#         for index, channel_name in enumerate(self.channels.name):
#             if channel_name in channel_names_oi:
#                 indices_type_oi.append(index)

#         # iterate through the indices_type_oi and create a new ChannelData object
#         channel_data_clean_type = ChannelData(
#             name=[self.channels.name[index] for index in indices_type_oi],
#             component=[self.channels.component[index] for index in indices_type_oi],
#             ch_type=[self.channels.ch_type[index] for index in indices_type_oi],
#             tracked_point=[
#                 self.channels.tracked_point[index] for index in indices_type_oi
#             ],
#             units=[self.channels.units[index] for index in indices_type_oi],
#         )
#         # create a new MotionData object with the given channel type
#         recording_data_clean_type = RecordingData(
#             name=self.name,
#             data=self.data[:, indices_type_oi],
#             sampling_frequency=self.sampling_frequency,
#             times=self.times,
#             channels=channel_data_clean_type,
#             start_time=self.start_time,
#             events=self.events,
#         )

#         return recording_data_clean_type


# @dataclass
# class MotionData:
#     """
#     A data class to hold meaningful groups of motion data entries.
#     For example, a recording of a participant walking on a treadmill with multiple
#     motion capture systems running.
#     Also can be a group of recordings from multiple participants performing the same task.

#     Attributes:
#         name (str): A name for the recording data.
#         data (np.ndarray): A nD numpy array of shape (n_samples, n_channels) containing
#             the motion data. Channels MUST have the same sampling frequency.
#         sampling_frequency (float): The sampling frequency of the motion data.
#         times (np.ndarray): A 1D numpy array of shape (n_samples,) containing the
#             timestamps of the motion data. If no time stamps are provided from the
#             system, timestamps are relative to the start of the recording.
#         channels (ChannelData): A ChannelData object containing information about the
#             channels in the data.
#         start_time (float): The start time of the recording in seconds. 0 if no time
#             stamps are provided from the system.
#         events (Optional[EventData]): An optional EventData object containing information
#             about events during the recording (default is None).
#     """

#     data: List[RecordingData]
#     times: np.ndarray  # Can be a 1D array representing timestamps
#     info: List[FileInfo]

#     @classmethod
#     def synchronise_recordings(self, systems: List[RecordingData]):
#         """
#         This functions uses the times provided from the systems to create a globally
#         valid times vector. If no start time is provided or if the start times start at
#         0, meaningful synchronization is not possible functions returns a warning.

#         Args:
#             systems (RecordingData): List of RecordingData objects

#         Returns:
#             self (RecordingData): A RecordingData object with a globally valid times vector

#         """

#         # find the start time of each system if not 0
#         start_times = []
#         for system in systems:
#             if system.start_time != 0:
#                 start_times.append(system.start_time)

#         # if all start times are 0, no meaningful synchronization is possible, give warning
#         if len(start_times) == 0:
#             print(
#                 "Warning: No start times provided or all start times are 0. No meaningful synchronization is possible."
#             )
#             return

#         # find the minimum start time and the index of the system with the minimum start time
#         min_start_time = min(start_times)
#         min_start_time_index = start_times.index(min_start_time)

#         # find the time difference between the start time of the system with the minimum start time and the other systems
#         time_diffs = []
#         for system in systems:
#             time_diffs.append(system.start_time - min_start_time)

#         # find the highest time stamp in regard to the system with the minimum start time
#         max_time = 0
#         for system in systems:
#             if system.times[-1] > max_time:
#                 max_time = system.times[-1]

#         # find the system with the highest sampling frequency
#         max_sampling_frequency = 0
#         for system in systems:
#             if system.sampling_frequency > max_sampling_frequency:
#                 max_sampling_frequency = system.sampling_frequency

#         # create a new times vector with the highest sampling frequency and the highest time stamp
#         new_times = np.arange(0, max_time, 1 / max_sampling_frequency)

#         # store new time vector
#         self.times = new_times

#         return
