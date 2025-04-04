# Tutorial: Calculation of Gait Spatio-temporal Parameters

**Author:** Masoud Abedinifar

**Last update:** Wed 04 April 2025

## Learning objectives
By the end of this tutorial, you will be able to:

- Load and process IMU-based gait event data (initial contacts, final contacts, and gait sequences).
- Apply the `GaitSpatioTemporalParameters` class to analyze gait using pre-detected event annotations.
- Compute **temporal gait parameters**, such as step time, stride time, stance, swing, single and double support times.
- Calculate **temporophasic parameters**, such as stance and swing times as a percentage of the gait cycle.
- Estimate **spatial gait parameters**, including step and stride lengths using the inverted pendulum model.
- Derive **spatio-temporal parameters**, such as stride speed for each foot.

## Gait Spatio-Temporal Parameters

This section demonstrates how to calculate clinically relevant gait parameters using the `GaitSpatioTemporalParameters` class from the toolbox. The algorithm uses pre-detected gait events (initial contacts, final contacts, and gait sequences) to extract `temporal`, `temporophasic`, `spatial`, and `spatio-temporal` metrics based on validated biomechanical definitions.

### Parameters Computed

- **Temporal Parameters**  
  Step time, stride time, stance time, swing time, single support time, and double support time  
  *(Based on: Moe-Nilssen et al. 2020 [`2`], Hollman et al. 2011 [`3`], Hass et al. 2012 [`4`])*

- **Temporophasic Parameters**  
  Stance, swing, single, and double support durations as a percentage of the gait cycle  
  *(Based on: Moe-Nilssen et al. 2020 [`2`], Hollman et al. 2011 [`3`])*

- **Spatial Parameters**  
  Step and stride lengths using the inverted pendulum model and double integration of vertical acceleration  
  *(Based on: Cerny et al. 2015 [`5`])*

- **Spatio-Temporal Parameters**  
  Stride speeds calculated by combining spatial and temporal metrics  
  *(Based on: Moe-Nilssen et al. 2020 [`2`], Hollman et al. 2011 [`3`], Hass et al. 2012 [`4`])*


References

[`1`] Zijlstra, W., & Hof, A. L. (2003). *Assessment of spatio-temporal gait parameters from trunk accelerations during human walking*. Gait & Posture.  

[`2`] Moe-Nilssen, R., Helbostad, J. L., et al. (2020). *Spatiotemporal gait parameters for older adults*. Gait & Posture. 

[`3`] Hollman, J. H., et al. (2011). *Normative spatiotemporal gait parameters in older adults*. Gait & Posture. 

[`4`] Hass, C. J., et al. (2012). *Quantitative normative gait data in a large cohort of ambulatory persons with Parkinson’s disease*. PLOS ONE.  

[`5`] Cerny, M., Noury, N., & Deplorte, L. (2015). *Validation of the inverted pendulum model for gait length calculation*. IEEE EMBC.

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
participant_id = "pp002"
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
                    0         0.984890       -0.011232        0.178212
                    1         0.985368       -0.012204        0.187495
                    2         0.987280       -0.007803        0.186044
                    3         0.987818       -0.008803        0.185045
                    4         0.986324       -0.007317        0.194328
                    ...       ...             ...             ...
                    1642      0.912108       -0.182120        0.357906
                    1643      0.920414       -0.195325        0.338374
                    1644      0.922864       -0.208986        0.336923
                    1645      0.930692       -0.210930        0.307141
                    1646      0.930692       -0.210930        0.311041

        [1647 rows x 3 columns]


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

        gyro_data (deg/s):      pelvis_GYRO_x  pelvis_GYRO_y  pelvis_GYRO_z
                    0           2.360255       1.661669      -0.263280
                    1           1.486229       0.962426       0.438799
                    2           1.312964       1.223030       0.000000
                    3           1.659495       1.573941       0.087760
                    4           0.785468       1.486213      -0.522744
                    ...         ...            ...            ...
                    1642        41.529716      -9.268191      -4.548252
                    1643        43.015945      -7.518795      -7.169601
                    1644        43.716705      -4.458640      -9.703192
                    1645        43.628147      -4.110309     -13.026620
                    1646        43.889973      -1.661669     -15.125226

        [1647 rows x 3 columns]

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


### Data Units and Conversion to SI Units

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
              onset     duration        event_type
        0     385       0               final_contact_left
        1     475       0               initial_contact_left
        2     505       0               final_contact_right
        3     574       0               start
        4     590       0               initial_contact_right
        5     625       0               final_contact_left
        6     712       0               initial_contact_left
        7     739       0               final_contact_right
        8     817       0               initial_contact_right
        9     843       0               final_contact_left
        10    923       0               initial_contact_left
        11    950       0               final_contact_right
        12   1028       0               initial_contact_right
        13   1056       0               final_contact_left
        14   1136       0               initial_contact_left
        15   1163       0               final_contact_right
        16   1242       0               initial_contact_right
        17   1274       0               final_contact_left
        18   1354       0               initial_contact_left
        19   1387       0               final_contact_right
        20   1426       0               stop
        21   1469       0               initial_contact_right
        22   1528       0               final_contact_left

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
        0       2.87      4.26        gait sequence   imu


### Extract Reference Initial and Final Contact Events

Initial and final contact events are extracted from the raw `events_df` and converted to a BIDS-compatible format for downstream gait analysis. All extracted contacts are combined into a single DataFrame and added to the `recording` object via the `add_events()` method.

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
            onset       duration        event_type          rl_label        tracking_system
        0   2.375       0.0             initial contact     left            imu
        1   2.950       0.0             initial contact     right           imu
        2   3.560       0.0             initial contact     left            imu
        3   4.085       0.0             initial contact     right           imu
        4   4.615       0.0             initial contact     left            imu
        5   5.140       0.0             initial contact     right           imu
        6   5.680       0.0             initial contact     left            imu
        7   6.210       0.0             initial contact     right           imu
        8   6.770       0.0             initial contact     left            imu
        9   7.345       0.0             initial contact     right           imu
        10  1.925       0.0             final contact       left            imu
        11  2.525       0.0             final contact       right           imu
        12  3.125       0.0             final contact       left            imu
        13  3.695       0.0             final contact       right           imu
        14  4.215       0.0             final contact       left            imu
        15  4.750       0.0             final contact       right           imu
        16  5.280       0.0             final contact       left            imu
        17  5.815       0.0             final contact       right           imu
        18  6.370       0.0             final contact       left            imu
        19  6.935       0.0             final contact       right           imu
        20  7.640       0.0             final contact       left            imu



## Load Gait Events and Initialize Spatio-Temporal Analysis

To begin the analysis, an instance of the `GaitSpatioTemporalParameters` class must be created. This object will store and process all relevant gait events, including:

- **Initial Contacts (IC)**  
- **Final Contacts (FC)**  
- **Gait Sequences**

The `detect()` method is used to load these pre-detected events from the recording. The input data should be structured as a DataFrame (typically extracted from `recording.events["imu"]`) where each event is labeled with an `event_type` such as `"initial contact"`, `"final contact"`, or `"gait sequence"`.

### Inputs to the `detect()` method:

- **`gait_sequences`** (`pd.DataFrame`)  
  DataFrame containing gait sequence events with the columns:
  - `onset`: Start time of the gait sequence (in seconds)
  - `duration`: Duration of the sequence (in seconds)

- **`initial_contacts`** (`pd.DataFrame`)  
  DataFrame containing initial contact events with:
  - `onset`: Time of initial contact (in seconds)
  - `rl_label`: Side label, either `"left"` or `"right"`

- **`final_contacts`** (`pd.DataFrame`)  
  DataFrame containing final contact events with:
  - `onset`: Time of final contact (in seconds)
  - `rl_label`: Side label, either `"left"` or `"right"`


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

### Calculation of Temporal Gait Parameters

Temporal gait parameters provide information about the timing of gait events such as initial contact (IC) and final contact (FC). These parameters are calculated using the `temporal_parameters()` method, based on pre-identified gait events within each gait sequence.

This method computes:

- **Step time [s]**: Time from IC of one foot to IC of the opposite foot.
- **Stride time [s]**: Time between two consecutive ICs of the same foot.
- **Stance time [s]**: Duration from IC to FC of the same foot.
- **Swing time [s]**: The difference between stride time and stance time.
- **Single support time [s]**: Time during stance when only one foot is on the ground.
- **Double support time [s]**: Time during which both feet are on the ground.

The results are stored in `gait_stp.step_temporal_parameters_` and `gait_stp.stride_temporal_parameters_` with one row per step and stride, respectively.

```python
# Compute temporal parameters (step + stride times)
gait_stp.temporal_parameters()
```

```python
# Display step-level temporal parameters
print("Step-level temporal gait parameters [s]:")
print(gait_stp.step_temporal_parameters_)
```
        Step-level temporal gait parameters [s]:
                gait_sequence_id    step_id     foot    onset   end_time    step_time
        0       0                   0           right   2.95    3.56        0.61
        1       0                   1           left    3.56    4.08        0.52
        2       0                   2           right   4.08    4.61        0.53
        3       0                   3           left    4.61    5.14        0.52
        4       0                   4           right   5.14    5.68        0.54
        5       0                   5           left    5.68    6.21        0.53
        6       0                   6           right   6.21    6.77        0.56


```python
# Display stride-level temporal parameters
print("\nStride-level temporal gait parameters [s]:")
print(gait_stp.stride_temporal_parameters_)
```


        Stride-level temporal gait parameters [s]:
                gait_sequence_id    stride_id   foot    onset   end_time    stride_time    stance_time   swing_time   single_support_time    double_support_time
        0       0                   0           right   2.95    4.08        1.13           0.74          0.39         0.43                   0.31     
        1       0                   1           left    3.56    4.61        1.05           0.65          0.40         0.40                   0.26   
        2       0                   2           right   4.08    5.14        1.00           0.66          0.39         0.40                   0.26   
        3       0                   3           left    4.61    5.68        1.06           0.66          0.40         0.39                   0.27    
        4       0                   4           right   5.14    6.21        1.07           0.67          0.39         0.40                   0.27    
        5       0                   5           left    5.00    6.77        1.09           0.69          0.40         0.39                   0.29   

      
            
### Calculation of Temporophasic Gait Parameters

After computing the temporal gait parameters, the `temporophasic_parameters()` method calculates percentage-based gait cycle phases.

These temporophasic parameters describe the **relative duration of each phase** within a stride for both legs, including:

- **Stance [%]**: Percentage of the stride where the foot is in contact with the ground.
- **Swing [%]**: Percentage of the stride where the foot is in the air.
- **Single support [%]**: Time during which only one foot is on the ground.
- **Double support [%]**: Time during which both feet are in contact with the ground.

All outputs are given as percentages and stored in `gait_stp.temporophasic_parameters_`, with one row per stride.

```python
# Compute temporophasic parameters (percent of stride)
gait_stp.temporophasic_parameters()

# Display temporophasic parameters
print("\nTemporophasic parameters per stride (% of gait cycle):")
print(gait_stp.temporophasic_parameters_)
```

        Temporophasic parameters per stride (% of gait cycle):
                gait_sequence_id    stride_id   foot    stance_pct    swing_pct    single_support_pct    double_support_pct  
        0       0                   0           right   65.64         34.36        38.33                 27.31    
        1       0                   1           left    62.09         37.91        36.97                 25.12    
        2       0                   2           right   63.03         36.97        37.91                 25.12     
        3       0                   3           left    62.44         37.56        36.62                 25.82     
        4       0                   4           right   63.08         36.92        37.38                 25.70     
        5       0                   5           left    63.30         36.70        36.24                 27.06      
               

### Calculation of Spatial Gait Parameters

After computing the temporal gait parameters, the `spatial_parameters()` method estimates spatial gait metrics using vertical acceleration data from a wearable IMU.

These spatial parameters describe the **distance covered** during each step and stride using a double integration approach and the inverted pendulum model. The method requires:

- `accel_data`: A DataFrame containing vertical acceleration signals.
- `v_acc_col_name`: Name of the column containing vertical acceleration values (e.g., `"pelvis_ACCEL_z"`).
- `sampling_freq_Hz`: The sampling frequency of the signal in Hz.
- `wearable_height`: The sensor height from the ground in meters (default: 1.0 m).

The method calculates:

- **Step length [m]**: Distance between two alternating foot contacts (e.g., left → right).
- **Stride length [m]**: Sum of two consecutive step lengths (left + right or right + left).

All outputs are in meters and stored as:

- `gait_stp.step_spatial_parameters_`: One row per step.
- `gait_stp.stride_spatial_parameters_`: One row per stride.

```python
# Estimate spatial parameters using vertical acceleration
# Ensure acceleration is vertical and in m/s^2
gait_stp.spatial_parameters(
    accel_data=accel_data,
    v_acc_col_name="pelvis_ACCEL_z",  # vertical axis column
    sampling_freq_Hz=sampling_frequency,
    wearable_height=1.0  # in meters
)
```

```python
# Display step-level spatial parameters
print("\nStep-level spatial gait parameters:")
print(gait_stp.step_spatial_parameters_)
```

        Step-level spatial gait parameters:
                gait_sequence_id    step_id     foot      step_length
        0       0                   0           right     0.59
        1       0                   1           left      0.50
        2       0                   2           right     0.51
        3       0                   3           left      0.50
        4       0                   4           right     0.52
        5       0                   5           left      0.51
        6       0                   6           right     0.53


```python
# Display stride-level spatial parameters
print("\nStride-level spatial gait parameters:")
print(gait_stp.stride_spatial_parameters_)
```


        Stride-level spatial gait parameters:
                gait_sequence_id    stride_id   foot    stride_length
        0       0                   0           right   1.09
        1       0                   1           left    1.01
        2       0                   2           right   1.01
        3       0                   3           left    1.02
        4       0                   4           right   1.03
        5       0                   5           left    1.04
         
### Calculation of Spatio-Temporal Gait Parameters

Once both temporal and spatial gait parameters are computed, the `spatiotemporal_parameters()` method calculates stride-level spatio-temporal metrics by combining stride time and stride length.

This method provides:

- **Stride speed [m/s]**: The walking speed during a stride, computed as `stride_length / stride_time`.

The result is stored in:

- `gait_stp.spatiotemporal_parameters_`: One row per stride including the stride speed per foot.

```python
# Compute spatiotemporal parameters
gait_stp.spatiotemporal_parameters()

# Print results
print("Spatial gait spatiotemporal parameters per gait sequence:")
print(gait_stp.spatiotemporal_parameters_)
```

        Spatial gait spatiotemporal parameters per gait sequence:
                gait_sequence_id    stride_id   foot    stride_speed
        0       0                   0           right   0.96
        1       0                   1           left    0.96
        2       0                   2           right   0.96
        3       0                   3           left    0.96
        4       0                   4           right   0.96
        5       0                   5           left    0.96