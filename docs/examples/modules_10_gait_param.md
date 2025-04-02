# Tutorial: Calculation of Gait Spatio-temporal Parameters

**Author:** Masoud Abedinifar

**Last update:** Wed 02 April 2025

## Learning Objectives
By the end of this tutorial:

- You can calculate spatio-temporal gait parameters from pre-detected gait events.

## Gait Spatio-temporal Parameters

This example can be referenced by citing the package.

The example demonstrates the application of the `GaitSpatioTemporalParameters` for calculating gait parameters from pre-detected gait events.

This implementation uses algorithms based on literature-reported definitions:
- Temporal parameters are calculated following clinical definitions and validated studies [`1`-`4`].
- Temporophasic percentages (stance and swing times) are derived as portions of the gait cycle [`2`,`3`].

**References**

[`1`] Zijlstra, W., & At L. Hof (2004). Assessment of spatio-temporal gait parameters from trunk accelerations during human walking

[`2`] Moe-Nilssen, R., et al. (2020). Spatiotemporal gait parameters for older adults. Gait & Posture.

[`3`] Hollman, J. H., et al. (2011). Normative spatiotemporal gait parameters in older adults. Gait & Posture.

[`4`] Hass, C. J., et al. (2012). Quantitative normative gait data in a large cohort of ambulatory persons with Parkinson’s disease. PLoS ONE.


## Import Libraries

The necessary libraries such as numpy, matplotlib.pyplot, and keepcontrol dataset. You also need to `GaitSpatioTemporalParameters` from kielmat modules. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the `kielmat` library and its dependencies if you haven't already.

```python
import numpy as np
import pandas as pd
from pathlib import Path
import os
from kielmat.datasets import keepcontrol
from kielmat.modules.gsd import GaitSpatioTemporalParameters
```

## Data Preparation

To implement the `GaitSpatioTemporalParameters` from KielMAT, we first load sample data from keepcontrol dataset.

```python
# Dataset path
dataset_path = Path(os.getcwd()) / "_keepcontrol"

# Fetch the dataset
keepcontrol.fetch_dataset(dataset_path)
```

In this example, we use `imu` as tracking_system and `pelvis` as tracked points.

```python
# In this example, we use "imu" as tracking_system and "pelvis" as tracked points.
tracking_sys = "imu"
tracked_points = {tracking_sys: ["pelvis"]}
```

The `keepcontrol.load_recording` function is used to load the data.

```python
# The 'keepcontrol.load_recording' function is used to load the data from the specified file_path
participant_id = "pp001"
task = "walkPreferred"

recording = keepcontrol.load_recording(
    dataset_path=dataset_path,
    id=participant_id,
    task=task,
    tracking_systems=[tracking_sys], 
    tracked_points=tracked_points
)
```
Get the `pelvis` acceleration data and the corresponding unit from the recording as:

```python
# Load pelvis acceleration data
accel_data = recording.data[tracking_sys][
    ["pelvis_ACCEL_x", "pelvis_ACCEL_y", "pelvis_ACCEL_z"]
]

# Get the acceleration data unit from the recording
accel_unit = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"].str.contains("ACCEL", case=False)
]["units"].iloc[0]

# Print acceleration data
print(f"accel_data ({accel_unit}): {accel_data}")
```

    accel_data (g):       pelvis_ACCEL_x  pelvis_ACCEL_y  pelvis_ACCEL_z

                0         0.994132        0.036127        0.115730
                1         0.984370        0.032706        0.106439
                2         0.991221        0.038586        0.109871
                3         0.990250        0.039067        0.106439
                4         0.996587        0.039067        0.119627
                ...       ...             ...             ...
                1784      0.909642       -0.163080        0.191894
                1785      0.892572       -0.167490        0.199224
                1786      0.893543       -0.167490        0.195792
                1787      0.889661       -0.174812        0.209471
                1788      0.886236       -0.181653        0.194346

      [1789 rows x 3 columns]


Get the `pelvis` gyro data and the corresponding unit from the recording as:


```python
# Load lower back gyro data
gyro_data = recording.data[tracking_sys][
    ["pelvis_GYRO_x", "pelvis_GYRO_y", "pelvis_GYRO_z"]
]

# Get the gyro data unit from the recording
gyro_unit = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"].str.contains("GYRO", case=False)
]["units"].iloc[0]

# Print gyro data
print(f"gyro_data ({gyro_unit}): {gyro_data}")
```

    gyro_data (deg/s):       pelvis_GYRO_x  pelvis_GYRO_y  pelvis_GYRO_z

                  0         -1.221909       3.059488      -0.962528
                  1         -1.746291       1.748709      -0.174418
                  2         -1.924383       2.097848      -1.048660
                  3         -2.275620       3.410132      -0.525407
                  4         -1.573146       2.185133      -0.437121
                  ...             ...            ...            ...
                  1784    -162.098358       0.174570     -57.529877
                  1785    -161.836166      -1.136209     -57.267174
                  1786    -161.400833      -3.671987     -58.229702
                  1787    -160.787399      -4.021127     -58.492405
                  1788    -159.822739      -6.469620     -58.404121

        [1789 rows x 3 columns]


Get the corresponding sampling frequency of the data directly from the recording

```python
# Get the corresponding sampling frequency directly from the recording
sampling_frequency = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"] == "pelvis_ACCEL_x"
]["sampling_frequency"].values[0]

# Print sampling frequency
print(f"sampling frequency: {sampling_frequency} Hz")
```

sampling frequency: 200 Hz


#### Data Units and Conversion to SI Units

All input data provided to the modules in this toolbox should adhere to SI units to maintain consistency and accuracy across analyses. This ensures compatibility with the underlying algorithms, which are designed to work with standard metric measurements.

If any data is provided in non-SI units (e.g., acceleration in g instead of m/s²), it is needed that the data to be converted into the appropriate SI units before using it as input to the toolbox. Failure to convert non-SI units may lead to incorrect results or misinterpretation of the output.

For instance:

- **Acceleration:** Convert from g to m/s².
- **Gyro:** Convert from rad/s to deg/s.

```python
# Check unit of acceleration data
if accel_unit in ["m/s^2"]:
    pass  # No conversion needed
elif accel_unit in ["g", "G"]:
    # Convert acceleration data from "g" to "m/s^2"
    accel_data *= 9.81
    # Update unit of acceleration
    accel_unit = ["m/s^2"]

# Check unit of gyro data
if gyro_unit in ["deg/s", "°/s"]:
    pass  # No conversion needed
elif gyro_unit == "rad/s":
    # Convert gyro data from "rad/s" to "deg/s"
    gyro_data = np.rad2deg(gyro_data)
    # Update unit of gyro
    gyro_unit = ["deg/s"]
```


### Load Reference Gait Sequences and Initial/Final Contact Events with Corresponding Information

In this step, we load the reference gait events from the `.tsv` file associated with the selected participant and task.  

```python
# Dataset path and locate the events file
dataset_path = Path(os.getcwd()) / "_keepcontrol"

motion_folder = dataset_path / f"sub-{participant_id}" / "motion"
events_file = motion_folder / f"sub-{participant_id}_task-{task}_events.tsv"

# Load the .tsv file
events_df = pd.read_csv(events_file, sep="\t")
print(f"Loaded events from file:\n{events_df}")
```

    Loaded events from file:

          onset     duration      event_type
    0     722       0             start
    1     751       0             final_contact_left
    2     834       0             initial_contact_left
    3     861       0             final_contact_right
    4     941       0             initial_contact_right
    5     966       0             final_contact_left
    6    1050       0             initial_contact_left
    7    1075       0             final_contact_right
    8    1152       0             initial_contact_right
    9    1178       0             final_contact_left
    10   1259       0             initial_contact_left
    11   1287       0             final_contact_right
    12   1363       0             initial_contact_right
    13   1392       0             final_contact_left
    14   1470       0             initial_contact_left
    15   1500       0             final_contact_right
    16   1580       0             initial_contact_right
    17   1612       0             final_contact_left
    18   1613       0             stop
    19   1684       0             initial_contact_left
    20   1730       0             final_contact_right


### Extract Reference Gait Sequences Events

The reference gait sequences are defined using `start` and `stop` event pairs. Each pair marks the beginning and end of a gait sequence. We convert the sample-based onsets to seconds using the extracted sampling frequency and create a BIDS-compatible format table.

```python
# Filter gait sequence events
start_stop_events = events_df[
    events_df["event_type"].isin(["start", "stop"])
].reset_index(drop=True)

# Extract gait sequences from pairs
gait_sequences = []

for i in range(0, len(start_stop_events), 2):
    start_sample = start_stop_events.loc[i, "onset"]
    stop_sample = start_stop_events.loc[i + 1, "onset"]

    onset_sec = start_sample / sampling_frequency
    duration_sec = (stop_sample - start_sample) / sampling_frequency

    gait_sequences.append({
        "onset": onset_sec,
        "duration": duration_sec,
        "event_type": "gait sequence",
        "tracking_system": tracking_sys
    })

# Convert to DataFrame
gait_sequences_df = pd.DataFrame(gait_sequences)

# Add gait sequence events to recording
recording.add_events(tracking_system=tracking_sys, new_events=gait_sequences_df)

# Display added gait sequences
print("Gait sequences:")
print(gait_sequences_df)
```

  Gait sequences:

            onset     duration    event_type      tracking_system
    0       3.61      4.455       gait sequence   imu


### Extract Reference Initial and Final Contact Events

Initial and final contact events are extracted and converted to BIDS-compatible format.

```python
# Extract Initial Contacts
initial_contacts = events_df[
    events_df["event_type"].str.startswith("initial_contact")
].copy()

# Extract labels
initial_contacts["rl_label"] = initial_contacts["event_type"].apply(
    lambda x: "left" if "left" in x else "right"
)

# Create event type 
initial_contacts["event_type"] = "initial contact"
initial_contacts["onset"] = initial_contacts["onset"] / sampling_frequency
initial_contacts["duration"] = 0.0
initial_contacts["tracking_system"] = tracking_sys
initial_contacts = initial_contacts[["onset", "duration", "event_type", "rl_label", "tracking_system"]]

# Extract Final Contacts
final_contacts = events_df[
    events_df["event_type"].str.startswith("final_contact")
].copy()

# Extract label
final_contacts["rl_label"] = final_contacts["event_type"].apply(
    lambda x: "left" if "left" in x else "right"
)

# Create event type 
final_contacts["event_type"] = "final contact"
final_contacts["onset"] = final_contacts["onset"] / sampling_frequency
final_contacts["duration"] = 0.0
final_contacts["tracking_system"] = tracking_sys
final_contacts = final_contacts[["onset", "duration", "event_type", "rl_label", "tracking_system"]]

# Combine and add to recording
all_contact_events = pd.concat([initial_contacts, final_contacts], ignore_index=True)
recording.add_events(tracking_system=tracking_sys, new_events=all_contact_events)

# Print to verify
print("Initial and final contact events:")
print(all_contact_events)
```

    Initial and final contact events:

          onset       duration      event_type        rl_label      tracking_system
      0   4.170       0.0           initial contact   left          imu
      1   4.705       0.0           initial contact   right         imu
      2   5.250       0.0           initial contact   left          imu
      3   5.760       0.0           initial contact   right         imu
      4   6.295       0.0           initial contact   left          imu
      5   6.815       0.0           initial contact   right         imu
      6   7.350       0.0           initial contact   left          imu
      7   7.900       0.0           initial contact   right         imu
      8   8.420       0.0           initial contact   left          imu
      9   3.755       0.0           final contact     left          imu
      10  4.305       0.0           final contact     right         imu
      11  4.830       0.0           final contact     left          imu
      12  5.375       0.0           final contact     right         imu
      13  5.890       0.0           final contact     left          imu
      14  6.435       0.0           final contact     right         imu
      15  6.960       0.0           final contact     left          imu
      16  7.500       0.0           final contact     right         imu
      17  8.060       0.0           final contact     left          imu
      18  8.650       0.0           final contact     right         imu



## Calculate Gait Spatio-Temporal Parameters

Now, we are running `GaitSpatioTemporalParameters` to calculate gait spatio-temporal parameters using gait events information.

Inputs of the class are as follows:

- **`gait_sequences`** (`pd.DataFrame`):  
  Gait sequence events with onset and duration.

- **`initial_contacts`** (`pd.DataFrame`):  
  Initial contacts with onset and rl_label.

- **`final_contacts`** (`pd.DataFrame`):  
  Final contacts with onset and rl_label.

```python
# Create an instance of the GaitSpatioTemporalParameters
gait_stp = GaitSpatioTemporalParameters()

# Call the spatio-temporal parameter calculation using gait_stp.detect
gait_stp.detect(
    gait_sequences=recording.events["imu"][recording.events["imu"]["event_type"] == "gait sequence"],
    initial_contacts=recording.events["imu"][recording.events["imu"]["event_type"] == "initial contact"],
    final_contacts=recording.events["imu"][recording.events["imu"]["event_type"] == "final contact"]
)
```

### Calculation of the Temporal Parameters

Next, the temporal gait parameters could be extracted using the `temporal_parameters`. The outputs are stored in the `temporal_df`.

```python
# Compute temporal parameters
temporal_df = gait_stp.temporal_parameters()

# Print result
print("Temporal gait parameters per gait sequence:")
print(temporal_df.temporal_parameters_)
```
    Temporal gait parameters per gait sequence:
    
                    gait_sequence_id                step_time_l                     step_time_r                 stride_time_l               stride_time_r               swing_time_l                    swing_time_r                stance_time_l                   stance_time_r           cadence  
        0           0                               [0.535, 0.51, 0.52, 0.55]       [0.545, 0.535, 0.535]       [1.08, 1.045, 1.055]        [1.055, 1.055, 1.085]       [0.415, 0.42, 0.405, 0.39]      [0.4, 0.385, 0.38, 0.4]     [0.66, 0.64, 0.665, 0.71]       [0.67, 0.675, 0.685]    128.69

            
### Calculation of the Temporophasic Parameters

Next, the temporophasic gait parameters could be extracted using the `temporophasic_parameters`. The outputs are stored in the `temporophasic_parameters_`.


```python
# Compute temporophasic parameters
phasic_df = gait_stp.temporophasic_parameters()

# Print result
print("Temporophasic gait parameters per gait sequence:")
print(phasic_df.temporophasic_parameters_)
```

    Temporophasic gait parameters per gait sequence:

                    gait_sequence_id                stance_time_pct_gc_l                stance_time_pct_gc_r                swing_time_pct_gc_l                swing_time_pct_gc_r 
        0           0                               [61.11, 61.24, 63.03]               [63.51, 63.98, 63.13]               [38.89, 38.76, 36.97]              [36.49, 36.02, 36.87]   