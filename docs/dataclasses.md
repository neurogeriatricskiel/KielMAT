In the following the NGMT dataclasses are described.
The dataclasses are used to store motion data in a standardized way. We provide some small set of import functions, each of which returns a dataclass.
User should easily be able to write their own import functions, to get the their data into the provided dataclasses (this step might take some thinking).
After the data is in the dataclasses, running functions on the data from our toolbox should be really straight forward.

## Relation of data classes
```mermaid
classDiagram
    class RawData {
        FilePath: str
        FileName: str
    }

    class FileInfo {
        SubjectID: str
        TaskName: str
        DatasetName: str
        FilePath: str
    }

    class ChannelData {
        name: List[str]
        component: List[str]
        ch_type: List[str]
        tracked_point: List[str]
        units: List[str]
        get_channel_units()
    }

    class EventData {
        name: List[str]
        onset: List[float]
        duration: List[float]
        trial_type: Optional[List[str]] = None
    }

    class RecordingData {
        name: str
        data: np.ndarray
        sampling_frequency: float
        times: np.1darray
        channels: ChannelData
        start_time: float
        types: List[str]
        GSD()
        ICD()
    }

    class MotionData {
        data: List[RecordingData]
        times: np.ndarray  # Can be a 1D array representing timestamps
        info: List[FileInfo]
        ch_names: List[str]  # Can be a list of channel names
        synchronise_recordings(self, systems: List[RecordingData]):
    }

    RecordingData --> MotionData: raw data with same sampling rate
    ChannelData --> RecordingData: info per channel
    EventData --> RecordingData: info about potential events
    FileInfo --> MotionData: indent on disk
    FileInfo --> ChannelData: info per channel 
    FileInfo --> RecordingData: raw time series data
    FileInfo --> EventData: Include events if available in raw data
    RawData --> FileInfo: Get info from file
```

This is the planned class structure for motion data. Data from any file format can ultimately be imported into the `MotionData` class. The `MotionData` object contains a `FileInfo` object. The `FileInfo` object contains information about the file, such as the subject ID, the task name, the project name and the file path. The `MotionData` class also contains a list of `RecordingData` objects. 

Each `RecordingData` object contains the raw data, the sampling rate, the time stamps and the channel info (`ChannelData`) of a tracking system. It is up to the user how to group the source data into a tracking system.
The `RecordingData` object can also contain information about events. The `EventData` objects stores information about events such as onset or duration.

The `ChannelData` object is used to store the channel name, the channel type, the channel units and the tracked point.

::: utils.data_classes