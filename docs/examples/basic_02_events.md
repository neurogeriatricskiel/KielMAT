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
# load the data from mobilised dataset
recording = mobilised.load_recording()

# specify which tracking system you want to use
tracking_sys = 'SU'
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

    36 gait sequence(s) detected.
    


```python
# Add events to the recording as a dictionary including tracking system and events
gait_sequence_events = gsd.gait_sequences_
recording.add_events(tracking_system=tracking_sys, new_events=gait_sequence_events)
print(f"events: {recording.events}")
```

    events: {'SU':        onset  duration     event_type tracking_system
    0     22.650    17.075  gait sequence            None
    1     49.150     7.475  gait sequence            None
    2     97.025   120.400  gait sequence            None
    3    229.550     9.225  gait sequence            None
    4    247.900    29.075  gait sequence            None
    5    296.225   189.600  gait sequence            None
    6    490.300    25.575  gait sequence            None
    7    562.925    15.075  gait sequence            None
    8    581.900    18.875  gait sequence            None
    9    607.050    56.600  gait sequence            None
    10   667.325   101.900  gait sequence            None
    11   784.500    42.775  gait sequence            None
    12   835.675   174.675  gait sequence            None
    13  1034.900    42.050  gait sequence            None
    14  1103.075    39.475  gait sequence            None
    15  1153.750    13.125  gait sequence            None
    16  1184.900     5.775  gait sequence            None
    17  1219.175    21.225  gait sequence            None
    18  1244.450    40.675  gait sequence            None
    19  1480.025     5.250  gait sequence            None
    20  1500.625    47.275  gait sequence            None
    21  1582.600    13.375  gait sequence            None
    22  1605.600    10.700  gait sequence            None
    23  1624.700    36.275  gait sequence            None
    24  1674.075     6.700  gait sequence            None
    25  5301.850     9.525  gait sequence            None
    26  5412.575    10.500  gait sequence            None
    27  5481.150    12.550  gait sequence            None
    28  5498.500     6.500  gait sequence            None
    29  5528.475    23.200  gait sequence            None
    30  5593.175    39.650  gait sequence            None
    31  5676.900    13.200  gait sequence            None
    32  5723.425    32.125  gait sequence            None
    33  5770.050    13.575  gait sequence            None
    34  5796.100     6.700  gait sequence            None
    35  6762.300   125.400  gait sequence            None}
    

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
recording.export_events(file_path = r'../examples/data', file_name='gait_sequence.csv')
```
