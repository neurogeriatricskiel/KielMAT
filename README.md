# NGMT

## NeuroGeriatricMotionToolbox

Python based toolbox for processing motion data

## Structure

```markdown
│ngmt                <- Main folder. It is initialized as a Git
│                       repository with a reasonable .gitignore file.
│
├── examples         <- Various scripts, e.g. simulations, plotting, analysis,
│                       The scripts use the `ngmt` folder for their base code.
|
├── info             <- Slides, Paper, basically any document related to the work in progress.
│
├── ngmt             <- Source code for use in this project. Contains functions,
│                       structures and modules that are used throughout
│                       the project and in multiple scripts.
│
├── test             <- Folder containing tests for `ngmt`.
│
├── README.md        <- Top-level README. 
├── LICENSE
├── requirements.txt <- The requirements file for reproducing the analysis environment, e.g.
│                       generated with `pip freeze > requirements.txt`. Might be replaced by
│                       a `environment.yml` file for Anaconda.
├── setup.py         <- makes project pip installable (pip install -e .) so src can be imported
|
└── .gitignore       <- focused on Python VS Code
```

## Realation of data classes
```mermaid
classDiagram
    class MotionData {
        info: FileInfo
        channels: ChannelMetaData
        times: np.ndarray
        time_series: np.ndarray
        check_channel_info()
        get_inital_contacts()
    }

    class FileInfo {
        SubjectId: str
        TaskName: str
        SamplingFrequency: float
        FilePath: str
        import_data()
    }

    class ChannelMetaData {
        name: list[int]
        component: list[str]
        ch_type: list[str]
        tracked_point: list[int]
        units: list[int]
        get_channel_units(): str
    }

    class DatasetInfo {
        SubjectIds: list[str]
        TaskNames: list[str]
        group_data()
    }

    MotionData <-- FileInfo: indent on disk
    MotionData <-- ChannelMetaData: info per channel in python
    DatasetInfo <-- MotionData: info per dataset
    FileInfo --> ChannelMetaData: info per channel on disk
```

## Authors

[Masoud Abedinifar](https://github.com/masoudabedinifar), [Robbin Romijnders](https://github.com/rmndrs89) & [Julius Welzel](https://github.com/JuliusWelzel)
