# Tutorial: Events in data classes

**Author:** Masoud Abedinifar & Julius Welzel

**Last update:** Thu 14 Mar 2024

## Learning objectives
By the end of this tutorial:

- Load data from a recording that belongs to one of the available datasets.
- Store events from algorithms in the datclass per recording file.
- Export [events to BIDS format](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html).

## Import libraries
The necessary libraries such as numpy, matplotlib.pyplot, dataset (mobilised), Paraschiv-Ionescu gait sequence detection, and Paraschiv-Ionescu initial contact detection algorithms are imported from their corresponding modules. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the 'kielmat' library and its dependencies if you haven't already.


```python
from kielmat.datasets import mobilised
from kielmat.modules.gsd import ParaschivIonescuGaitSequenceDetection
```

First load the data and put in the desired dataclasses.


```python
# The 'file_path' variable holds the absolute path to the data file
file_path = (
    r"..\examples\data\exMobiliseFreeLiving.mat"
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
    

The events are put into a pandas DataFrame, and follow the conventions outlined in the BIDS documentation (i.e. https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html).

### Gait sequence events in dataclass


```python
# Create an instance of the ParaschivIonescuGaitSequenceDetection class
gsd = ParaschivIonescuGaitSequenceDetection()

# Call the gait sequence detection using gsd.detect to detect gait sequences
gsd = gsd.detect(
    data=acceleration_data, sampling_freq_Hz=sampling_frequency, plot_results=False
)
```

    72 gait sequence(s) detected.
    


```python
# Add events to the recording as a dictionary including tracking system and events
gait_sequence_events = gsd.gait_sequences_
recording.add_events(tracking_system=tracking_sys, new_events=gait_sequence_events)
print(f"events: {recording.events}")
```

    events: {'SU':         
        onset       duration    event_type         tracking_system
    0      17.450     6.525     gait sequence      None
    1      96.500     5.350     gait sequence      None
    2     145.150     7.500     gait sequence      None
    3     451.425    21.375     gait sequence      None
    4     500.700     6.775     gait sequence      None
    ..        ...       ...               ...       ...
    67   9965.875    10.700     gait sequence      None
    68  10035.875    11.700     gait sequence      None
    69  10078.075    18.575     gait sequence      None
    70  10251.475     8.925     gait sequence      None
    71  10561.200    11.325     gait sequence      None
    
    [72 rows x 4 columns]}
    

### Store events to events.tsv file following the BIDS convention

Add some information about the recording first which is necessary for the BIDS file name convention.
KielMAT has some implemented check on the information to make sure that the file name is in the correct format.


```python
recording.add_info("Subject", "CHF01")
recording.add_info("Task", "walking_outside")
```

    Warning: The value of the key 'Subject' should be lower case. Converted to lower case.
    Warning: The value of the key 'Task' should be lower case. Converted to lower case.
    Warning: The value of the key 'Task' should not contain underscore or space. Removed underscore and space.
    

Please notice that we a not to strict with the user. We just give a warning if the file name is not in BIDS like format. However, the user can still continue with the process.
But you better believe that the BIDS police will come and get you if you don't follow the rules.

Now as we have the events in the dataclass, we can export them to a [BIDS compatible events](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/task-events.html) file.


```python
recording.export_events(file_path = r'../examples/data', file_name='gait_sequence.csv', bids_compatible=True)
```
