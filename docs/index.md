# Welcome 

Welcome to the NeuroGeriatricMotionToolbox (NGMT). We are a Python based toolbox for processing motion data.

> The toolbox is currently under development and is not yet ready for use.

The toolbox is aimed at a wide variety of motion researchers who want to use open souce software to process their data.
We have implemented a wide variety of functions to process motion data, such as:

-   Gait sequence detection (GSD)
-   Inital contact detection (ICD)
-   More to follow ...

The idea is that various motion data can be loaded into our dedicated dataclasses which rely on principles from the [Motion-BIDS](https://bids-specification.readthedocs.io/en/latest/modality-specific-files/motion.html) standard.

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
        pick_channel_types()
    }

    class MotionData {
        data: List[RecordingData]
        times: np.ndarray  # Can be a 1D array representing timestamps
        info: List[FileInfo]
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

## Documentation
The full documentation can be found here.

## Installation
After the first release, the toolbox can be installed via pip

For now, the toolbox can be installed via the following steps:
1. Clone the repository
2. Create a virtual environment
3. Install the requirements
4. Install the toolbox

```bash
git clone https://github.com/neurogeriatricskiel/NGMT.git
cd NeuroGeriatricMotionToolbox
python -m venv venv_ngmt
source venv_ngmt/bin/activate
pip install -r environment.yml
pip install -e .
```

## Authors

[Masoud Abedinifar](https://github.com/masoudabedinifar), [Clint Hansen](mailto:c.hansen@neurologie.uni-kiel.de), [Walter Maetzler](mailto:w.maetzler@neurologie.uni-kiel.de), [Robbin Romijnders](https://github.com/rmndrs89) & [Julius Welzel](https://github.com/JuliusWelzel)
