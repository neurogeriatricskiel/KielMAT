from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


@dataclass(kw_only=True)
class NGMTRecording:
    """Dataclass to hold any data and associated infos for a NGMT recording.

    Attributes:
        data (dict): The data is stored as a pandas DataFrame for each unique tracking system.
        channels (dict): The channels descriptions are stored as a pandas DataFrame for each unique tracking system.
        events (dict): The events are stored as a pandas DataFrame for each unique tracking system.
        info (dict): The infos on the subject, task, and more, are stored as a nested dictionary.
        events_info (dict): The event infos are stored as a nested dictionary.
    """

    data: dict[str, pd.DataFrame]
    channels: dict[str, pd.DataFrame]
    info: None | dict[str, Any] = None
    events: None | dict[str, pd.DataFrame] = None
    events_info: None | dict[str, Any] = None

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

    def export_events(self, tracking_system: Optional[str] = None, file_path: str) -> None:
        """Export events for a specific tracking system to a file.

        Args:
            tracking_system (Optional[str]): Tracking system for which events are to be exported.
                If None, events from all tracking systems will be exported (default is None).
            file_path (str): Path to the file where events are to be exported.
        """
        if self.events is not None:
            if tracking_system is None:
                all_events = pd.concat(self.events.values(), keys=self.events.keys(), names=['tracking_system'])
                all_events.to_csv(file_path, index=False)
            elif tracking_system in self.events:
                self.events[tracking_system].to_csv(file_path, index=False)
                
# @dataclass
# class FileInfo:
#     """
#     A data class representing information about a file to be processed.
#     The main puprose is to store the metadata of the file for a larger dataset,
#     and the path to the file in the file system.

#     Attributes:
#         SubjectID (str): The identifier of the subject associated with the file.
#         TaskName (str): The name of the task or experiment associated with the file.
#         DatasetName (str): The name of the dataset to which the file belongs.
#         FilePath (str): The path to the file in the file system.
#     """

#     SubjectID: str
#     TaskName: str
#     DatasetName: str
#     FilePath: str


# @dataclass
# class ChannelData:
#     """
#     A data class for representing information about channels used in a recording.

#     Attributes:
#         name (List[str]): A list of channel names.
#         component (List[str]): A list of channel components.
#         ch_type (List[str]): A list of channel types.
#         tracked_point (List[str]): A list of tracked points.
#         units (List[str]): A list of measurement units.
#         placement (Optional[List[str]]): An optional list of placement information (default is None).
#         description (Optional[List[str]]): An optional list of channel descriptions (default is None).
#         range(Optional[List[float]]): An optional list of sensor ranges (default is None).
#         sampling_frequency (Optional[float]): An optional sampling frequency (default is None).
#         status (Optional[List[float]]): An optional list of channel statuses (default is None).
#         status_description (Optional[List[str]]): An optional list of status descriptions (default is None).

#     Raises:
#         ValueError: If the provided 'ch_type' is not one of the valid channel types.
#         ValueError: If the provided 'component' is not one of the valid component types.
#     """

#     name: List[str]
#     component: List[str]
#     ch_type: List[str]
#     tracked_point: List[str]
#     units: List[str]
#     placement: Optional[List[str]] = None
#     description: Optional[List[str]] = None
#     range: Optional[List[float]] = None
#     sampling_frequency: Optional[float] = None
#     status: Optional[List[float]] = None
#     status_description: Optional[List[str]] = None

#     # check if all entries in the list of self.ch_type are valid
#     def __post_init__(self):
#         for ch_type in self.ch_type:
#             if ch_type not in VALID_CHANNEL_TYPES:
#                 raise ValueError(
#                     f"Invalid channel type '{ch_type}'. Valid channel types are: {VALID_CHANNEL_TYPES}"
#                 )

#         for component in self.component:
#             if component not in VALID_COMPONENT_TYPES:
#                 raise ValueError(
#                     f"Invalid component '{component}'. Valid components are: {VALID_COMPONENT_TYPES}"
#                 )


# @dataclass
# class EventData:
#     """
#     A data class to describe timing and other properties of events during a recording.
#     Events can include stimuli presented to the participant,
#     participant responses or labeling of data samples.
#     Events can overlap in time.

#     Attributes:
#         name (List[str]): A list of event names.
#         onset (List[float]): A list of event onset times.
#         duration (List[float]): A list of event durations.
#         event_type (Optional[Union[str, List[str]]]): A list of event types.
#     """

#     def add_events(
#         self,
#         onset: Union[int, Sequence[int]],
#         duration: Union[int, Sequence[int]],
#         event_type: Union[str, Sequence[str]],
#         name: Optional[Union[str, Sequence[str]]],
#     ):
#         """
#         This function adds events to the EventData object.

#         Args:
#             name (str): name of the event
#             onset (float): onset of the event
#             duration (float): duration of the event

#         Returns:
#             self (EventData): An EventData object with the added event
#         """

#         # check if name is already present in the EventData object
#         if name in self.name:
#             raise ValueError(
#                 f"Event with name '{name}' already present in EventData object."
#             )

#         self.name.extend(name)
#         self.onset.extend(onset)
#         self.duration.extend(duration)

#         # sort the events by onset
#         # get indices of ascending order of onset
#         indices = np.argsort(self.onset)
#         # sort the lists by the indices
#         self.name = [self.name[i] for i in indices]
#         self.onset = [self.onset[i] for i in indices]
#         self.duration = [self.duration[i] for i in indices]

#         return self


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

#     def plot_events(
#         self,
#         event_types_oi,
#         axes=None,
#     ):
#         """
#         This function plots the events of a given event type.

#         Parameters:
#             event_types_oi (str): event type

#         Returns:
#             None
#         """

#         # find the indices_type_oi of the events with the given event type
#         indices_type_oi = []
#         for index, event_type in enumerate(self.events.name):
#             # check if event_types_oi is single string or list of strings of length > 1
#             if isinstance(event_types_oi, str):
#                 if event_type in event_types_oi:
#                     indices_type_oi.append(index)

#             elif isinstance(event_types_oi, list) and len(event_types_oi) > 1:
#                 for event_type_oi in event_types_oi:
#                     if event_type_oi in event_type:
#                         indices_type_oi.append(index)

#         # get the times of events
#         times_events = [self.events.onset[i] for i in indices_type_oi]

#         ax = axes if axes else plt.gca()

#         # plot the raw data
#         ax.plot(self.times, self.data, linewidth=1)

#         # plot the events
#         for time_event in times_events:
#             ax.axvline(time_event, color="black", linewidth=1)
#         ax.set_xlabel("Time [s]")

#         return

#     def detect_gait_sequence(self, sensor_id):
#         # check if sensor_id is a list of strings of length > 0
#         if isinstance(sensor_id, list) and len(sensor_id) > 0:
#             pass
#         else:
#             raise ValueError(f"sensor_id should be a list of strings of length > 0.")

#         # select acceleration data from recording
#         acceleration_data = self.pick_channel_types(channel_type_oi="ACCEL")
#         # select only lower back
#         acceleration_data_lower_back = acceleration_data.pick_channels(
#             channel_names_oi=sensor_id
#         )
#         # get sampling frequency
#         sampling_frequency = acceleration_data.sampling_frequency

#         # Use Gait_Sequence_Detection to detect gait sequence
#         gait_sequences = GSDB.Gait_Sequence_Detection(
#             imu_acceleration=acceleration_data_lower_back,
#             sampling_frequency=sampling_frequency,
#             plot_results=False,
#         )

#         # Add gait sequences to EventData object
#         self.events.add_events(
#             onset=gait_sequences["onset"],
#             duration=gait_sequences["duration"],
#             event_type="gait_sequence",
#             name="gait_sequence",
#         )

#         return self


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
