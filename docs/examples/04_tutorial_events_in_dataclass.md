# Tutorial: events in data classes

**Author (s):** Masoud Abedinifar & Julius Welzel

**Last update:** Thu 14 Mar 2024

## Learning objectives
By the end of this tutorial:

- Load data from a recording that belongs to one of the available datasets.
- Store events from algorithms in the datclass per recording file.
- Export [events to BIDS format](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html).

## Import Libraries
The necessary libraries such as numpy, matplotlib.pyplot, dataset (mobilised), and Paraschiv-Ionescu gait sequence detection algorithm are imported from its corresponding module. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the `ngmt` library and its dependencies if you haven't already.


```python
from ngmt.datasets import mobilised
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
```

First load the data and put in the desired dataclasses.


```python
# The 'file_path' variable holds the absolute path to the data file
file_path = (
    r"..\examples\data\chfDataMobilise.mat"
)

# In this example, we use "SU" as tracking_system and "LowerBack" as tracked points.
tracking_sys = "SU"
tracked_points = {tracking_sys: ["LowerBack"]}

# The 'mobilised.load_recording' function is used to load the data from the specified file_path
recording = mobilised.load_recording(
    file_name=file_path, tracking_systems=[tracking_sys], tracked_points=tracked_points
)
```

```python
# Load lower back acceleration data
acceleration_data = recording.data[tracking_sys][
    ["LowerBack_ACCEL_x", "LowerBack_ACCEL_y", "LowerBack_ACCEL_z"]
]
```

```python
# Get the corresponding sampling frequency directly from the recording
sampling_frequency = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"] == "LowerBack_ACCEL_x"
]["sampling_frequency"].values[0]
print(f"Sampling frequency: {sampling_frequency} Hz")
```
    Sampling frequency: 100.0 Hz

The events are put into a pandas DataFrame, and follow the conventions outlined in the BIDS documentation (i.e. [https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html)).


### Gait sequence events in dataclass

```python
# Create an instance of the ParaschivIonescuGaitSequenceDetection class
gsd = ParaschivIonescuGaitSequenceDetection(target_sampling_freq_Hz=40)

# Call the gait sequence detection using gsd.detect to detect gait sequences
gsd = gsd.detect(
    data=acceleration_data, sampling_freq_Hz=sampling_frequency, plot_results=False
)
```

    86 gait sequence(s) detected.

# Add events to the recording as a dictionary including tracking system and events

```python
gait_sequence_events = gsd.gait_sequences_
recording.add_events(tracking_system=tracking_sys, new_events=gait_sequence_events)
print(f"events: {recording.events}")
```
events:

    {'SU': onset    duration    event_type      tracking_systems  tracked_points
    0      4.5      5.25        gait sequence   SU                LowerBack
    1      90.22    10.3        gait sequence   SU                LowerBack
    2      106.07   5.6         gait sequence   SU                LowerBack
    3      116.22   10.35       gait sequence   SU                LowerBack
    4      141.27   5.85        gait sequence   SU                LowerBack
    ..     ...      ...         ...             ...               ...
    81     7617.15  4.15        gait sequence   SU                LowerBack
    82     7679.42  10.62       gait sequence   SU                LowerBack
    83     8090.62  4.2         gait sequence   SU                LowerBack
    84     8149.85  5.05        gait sequence   SU                LowerBack
    85     8184.87  21.45       gait sequence   SU                LowerBack
    
    [86 rows x 5 columns]}


### Store events to events.tsv file following the BIDS convention

Add some information about the recording first which is necessary for the BIDS file name convention. NGMT has some implemented check on the information to make sure that the file name is in the correct format.

```python
recording.add_info("Subject", "CHF01")
recording.add_info("Task", "walking_outside")
```

    Warning: The value of the key 'Subject' should be lower case. Converted to lower case.
    Warning: The value of the key 'Task' should be lower case. Converted to lower case.
    Warning: The value of the key 'Task' should not contain underscore or space. Removed underscore and space.

```python
recording.export_events(file_path = r'../examples/data', file_name='gait_sequence.csv', bids_compatible=True)
```