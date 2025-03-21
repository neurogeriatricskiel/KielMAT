# Tutorial: McCamley Initial Contact Classification

**Author:** Masoud Abedinifar

**Last update:** Fri 21 March 2025

## Learning Objectives
By the end of this tutorial:

- You can load data from [`Mobilized`](../datasets/mobilised.md), which is one of the available datasets.
- Apply the [`McCamley Initial Contact Classification`](../modules/ic_rlc_mccamley.md) algorithm to detect initial contact laterality.
- Analyze the classified initial contacts for further gait analysis.

## McCamley Initial Contact Classification

This example can be referenced by citing the package.

The example demonstrates the application of the McCamley Initial Contact Classification algorithm for distinguishing left and right initial contacts using gyroscope data collected from a lower back IMU sensor. The classification algorithm is implemented using [`kielmat.modules.rlc._mccamley`](https://github.com/neurogeriatricskiel/KielMAT/blob/gait-spatiotemporal-parameters/kielmat/modules/rlc/_mccamley.py). This algorithm is based on the research of McCamley et al. [1].

This algorithm aims to classify initial contacts (ICs) using gyroscope data collected from a lower back inertial measurement unit (IMU) sensor. The core of the algorithm lies in the `detect` method, where ICs are classified based on gyroscope signals. The method first preprocesses the gyroscope data by removing biases and applying a Butterworth bandpass filter to improve signal quality. The algorithm then evaluates the sign of the filtered signal at each detected initial contact position to classify the laterality as either `left` or ``right``.

The algorithm accounts for variations in signal processing by allowing different gyroscope components to be used for classification:

- **Vertical**: Uses only the vertical gyroscope signal.
- **Anterior-Posterior**: Uses only the anterior-posterior gyroscope signal.
- **Combined**: Uses the difference between vertical and anterior-posterior gyroscope signals (`Combined` = `Vertical` - `Anterior-Posterior`), which enhances classification accuracy by capturing both rotational components simultaneously [2].

Originally, the algorithm relied on the vertical axis signal alone [1]. However, Ullrich et al. [2] demonstrated that incorporating both vertical and anterior-posterior signals enhances detection accuracy. Users can select their preferred signal type and obtain the results as `left` or `right` labels accordingly.

Once classified, the initial contacts are stored in a pandas DataFrame (`ic_rl_list_` attribute), including their `onset` times and corresponding `rl_label` (left/right classification).

### **References**
[1] **McCamley et al.** (2012). *An Enhanced Estimate of Initial Contact and Final Contact Instants of Time Using Lower Trunk Inertial Sensor Data.* Gait & Posture, 36(2), 318-320. [https://doi.org/10.1016/j.gaitpost.2012.02.019](https://doi.org/10.1016/j.gaitpost.2012.02.019)

[2] **Ullrich et al.** (2022). *Machine Learning-Based Distinction of Left and Right Foot Contacts in Lower Back Inertial Sensor Data Improves Gait Analysis Accuracy.* IEEE Transactions on Neural Systems and Rehabilitation Engineering. [https://doi.org/10.1109/EMBC46164.2021.9630653](https://doi.org/10.1109/EMBC46164.2021.9630653)


## Import Libraries
The necessary libraries such as numpy, matplotlib.pyplot, dataset and MacCamley Initial Contact Classification algortihm are imported. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the `kielmat` library and its dependencies if you haven't already.

```python
import numpy as np
import pandas as pd
import os
from kielmat.datasets import mobilised
from kielmat.modules.gsd import ParaschivIonescuGaitSequenceDetection
from kielmat.modules.icd import ParaschivIonescuInitialContactDetection
from kielmat.modules.rlc import MacCamleyInitialContactClassification
from pathlib import Path
```

## Data Preparation

To implement the MacCamley Initial Contact Classification algortihm, we load example data from publicly available on the Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7547125.svg)](https://doi.org/10.5281/zenodo.7547125). 

The participant was monitored for 2.5 hours in a real-world setting while performing various daily activities. Additionally, structured tasks such as outdoor walking, stair ascent/descent, and transitions between rooms were included in the assessment [`3`].

#### Refertences

[`3`] Mazzà, Claudia, et al. "Technical validation of real-world monitoring of gait: a multicentric observational study." BMJ open 11.12 (2021): e050785. http://dx.doi.org/10.1136/bmjopen-2021-050785

```python
# Set the dataset path
dataset_path = Path(os.getcwd()) / "_mobilised"

# Fetch and load the dataset
mobilised.fetch_dataset(dataset_path=dataset_path)

# In this example, we use "SU" as tracking_system and "LowerBack" as tracked points.
tracking_sys = "SU"
tracked_points = {tracking_sys: ["LowerBack"]}

# The 'mobilised.load_recording' function is used to load the data from the specified file_path
recording = mobilised.load_recording(
    cohort="HA",  # Choose the cohort
    file_name="data.mat", 
    dataset_path=dataset_path)
```

Load and print acceleration data and it corresponding unit:

```python
# Load lower back acceleration data
accel_data = recording.data[tracking_sys][
    ["LowerBack_ACCEL_x", "LowerBack_ACCEL_y", "LowerBack_ACCEL_z"]
]

# Get the acceleration data unit from the recording
accel_unit = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"].str.contains("ACCEL", case=False)
]["units"].iloc[0]

# Print acceleration data
print(f"accel_data ({accel_unit}): {accel_data}")

```
    
    accel_data (g):         LowerBack_ACCEL_x  LowerBack_ACCEL_y  LowerBack_ACCEL_z

                0           0.967090          -0.019868          -0.148579
                1           0.969861          -0.022828          -0.149823
                2           0.983489          -0.028706          -0.146259
                3           0.987963          -0.027181          -0.147658
                4           0.988685          -0.028820          -0.144034
                ...         ...                ...                ...
                429149      0.056096          -0.014986           0.979918
                429150      0.057578          -0.013656           0.984973
                429151      0.056573          -0.013776           0.983177
                429152      0.060932          -0.016998           0.976655
                429153      0.061734          -0.014374           0.983742

    [429154 rows x 3 columns]

Load and print gyro data and it corresponding unit:

```python
# Load lower back gyro data
gyro_data = recording.data[tracking_sys][
    ["LowerBack_GYRO_x", "LowerBack_GYRO_y", "LowerBack_GYRO_z"]
]

# Get the gyro data unit from the recording
gyro_unit = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"].str.contains("GYRO", case=False)
]["units"].iloc[0]

# Print gyro data
print(f"gyro_data ({gyro_unit}): {gyro_data}")
```

    gyro_data (deg/s):         LowerBack_GYRO_x  LowerBack_GYRO_y  LowerBack_GYRO_z

                    0          9.980728          3.266193          4.721061
                    1          10.072528         2.991051          4.984530
                    2          9.958097          2.040483          4.950383
                    3          9.797709          1.421440          5.488496
                    4          9.373945          0.859908          5.282844
                    ...        ...               ...               ...
                    429149    -0.286461         -0.710477         -0.641677
                    429150    -0.091684         -0.813586         -0.263551
                    429151    -0.240635         -0.607326         -0.126057
                    429152    -0.091671         -0.412533         -0.263559
                    429153     0.022918         -0.550039         -0.194806

    [429154 rows x 3 columns]


Load and print sampling frequency of the data:

```python
# Get the corresponding sampling frequency
sampling_frequency = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"] == "LowerBack_ACCEL_x"
]["sampling_frequency"].values[0]
print(f'Data was filtered at', sampling_frequency, f'Hz')
```
Data was filtered at 100.0 Hz

#### Conversion of Data Units to SI Units

All input data provided to the modules in this toolbox should adhere to SI units to maintain consistency and accuracy across analyses. This ensures compatibility with the underlying algorithms, which are designed to work with standard metric measurements.

If the provided data is in non-SI units (e.g., acceleration in g instead of m/s²), it must be converted to SI units before being used in the analysis. Failure to convert non-SI units may lead to incorrect results or misinterpretation of the output.

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

## Applying McCamley Initial Contact Classification Algorithm

Now, we are running the McCamley Initial Contact Classification algorithm from the [`KielMAT.kielmat.modules.rlc._mccamley.MacCamleyInitialContactClassification`](https://github.com/neurogeriatricskiel/KielMAT/blob/gait-spatiotemporal-parameters/kielmat/modules/rlc/_mccamley.py) module to classify the laterality of initial contacts detected in gait sequences. 

For this purpose, we first apply the **Paraschiv-Ionescu Gait Sequence Detection** algorithm to identify gait sequences using acceleration data. The gait sequences are detected using [`KielMAT.kielmat.modules.gsd._paraschiv.ParaschivIonescuGaitSequenceDetection`](https://github.com/neurogeriatricskiel/KielMAT/tree/main/kielmat/modules/gsd/_paraschiv.py).

```python
# Create an instance of the ParaschivIonescuGaitSequenceDetection class
gsd = ParaschivIonescuGaitSequenceDetection()

# Call the gait sequence detection using gsd.detect to detect gait sequences
gsd = gsd.detect(
    accel_data=accel_data, 
    sampling_freq_Hz=sampling_frequency, 
    tracking_system="LowerBack", 
    plot_results=False
)

# Add events to the recording as a dictionary including tracking system and events
recording.add_events(tracking_system="LowerBack", new_events=gsd.gait_sequences_)

# Filter only gait sequence events
gait_sequence_events = recording.events["LowerBack"][recording.events["LowerBack"]["event_type"] == "gait sequence"]

# Print the filtered gait sequences
print(f'Gait sequences and thier corresponding information:')
print(gait_sequence_events)
```

        13 gait sequence(s) detected.
        Gait sequences and thier corresponding information:
                    onset       duration        event_type          tracking_system
            0       1348.200    7.050           gait sequence       LowerBack
            1       1372.025    4.600           gait sequence       LowerBack
            2       1388.750    199.675         gait sequence       LowerBack
            3       1596.050    24.700          gait sequence       LowerBack
            4       1694.750    457.475         gait sequence       LowerBack
            5       2168.000    310.625         gait sequence       LowerBack
            6       2585.900    43.700          gait sequence       LowerBack
            7       2633.625    315.600         gait sequence       LowerBack
            8       2956.600    9.475           gait sequence       LowerBack
            9       3029.725    16.450          gait sequence       LowerBack
            10      3087.550    9.825           gait sequence       LowerBack
            11      3195.125    12.825          gait sequence       LowerBack
            12      3464.250    14.375          gait sequence       LowerBack


Next, we apply the **Paraschiv-Ionescu Initial Contact Detection** algorithm to detect initial contacts within these gait sequences using [`KielMAT.kielmat.modules.icd._paraschiv.ParaschivIonescuInitialContactDetection`](https://github.com/neurogeriatricskiel/KielMAT/tree/main/kielmat/modules/icd/_paraschiv.py). 


```python
# Now, use Paraschiv-Ionescu initial contact detection algortihm to find initial contacts within detected gait sequences.
icd = ParaschivIonescuInitialContactDetection()

# Call the initial contact detection using icd.detect
icd = icd.detect(
    accel_data=accel_data,
    gait_sequences=gsd.gait_sequences_,
    sampling_freq_Hz=sampling_frequency,
    tracking_system="LowerBack", 
    v_acc_col_name="LowerBack_ACCEL_x"
)

# Add events to the recording as a dictionary including tracking system and events
recording.add_events(tracking_system="LowerBack", new_events=icd.initial_contacts_)

# Filter only gait sequence events
initaal_contact_events = recording.events["LowerBack"][recording.events["LowerBack"]["event_type"] == "initial contact"]

# Print the filtered gait sequences
print(f"Initial contacts information:")
print(initaal_contact_events)
```

        Initial contacts information:
                onset       event_type          duration    tracking_systems

        0       1348.700    initial contact     0           LowerBack
        1       1349.350    initial contact     0           LowerBack
        2       1349.975    initial contact     0           LowerBack
        3       1350.525    initial contact     0           LowerBack
        4       1351.050    initial contact     0           LowerBack
        ...     ...         ...                 ...         ...
        2660    3475.750    initial contact     0           LowerBack
        2661    3476.300    initial contact     0           LowerBack
        2662    3476.825    initial contact     0           LowerBack
        2663    3477.375    initial contact     0           LowerBack
        2664    3477.925    initial contact     0           LowerBack

        [2665 rows x 4 columns]


Once the initial contacts are detected, the McCamley Initial Contact Classification algorithm is used to classify them as left or right based on gyroscope signals using the McCamley method.

### Inputs for McCamley Initial Contact Classification Algorithm:

- **`gyro_data`** (`pd.DataFrame`):  
  A DataFrame containing gyroscope signals (N, 3), typically including x, y, and z components.

- **`sampling_freq_Hz`** (`float`):  
  Sampling frequency of the gyroscope signal in Hertz (Hz).

- **`v_gyr_col_name`** (`str`):  
  Name of the column in `gyro_data` that corresponds to the **vertical gyroscope signal**.

- **`ap_gyr_col_name`** (`str`):  
  Name of the column in `gyro_data` that corresponds to the **anterior-posterior gyroscope signal**.

- **`ic_timestamps`** (`pd.DataFrame`):  
  A DataFrame containing detected initial contact (IC) timestamps. Must include an `onset` column with timestamps in seconds.

- **`signal_type`** (`str`, optional, default = `"vertical"`):  
  Determines which gyroscope signal to use for left/right classification:
  
  - **`"vertical"`**: Uses the **vertical** gyroscope signal.
  - **`"anterior_posterior"`**: Uses the **anterior-posterior** gyroscope signal.
  - **`"combined"`**: Uses a difference-based signal: `vertical - anterior_posterior`.

- **`recording`** (`KielMATRecording`, optional):  
  If provided, the output labels (`rl_label`) will be added directly to the `recording.events` table.

- **`tracking_system`** (`str`, required if `recording` is given):  
  The tracking system key in the recording (e.g., `"LowerBack"`) used to locate and update initial contact events.


```python
# Initialize McCamley classifier
mccamley_ic_classifier = MacCamleyInitialContactClassification()

# Apply classification
mccamley_ic_classifier.detect(
    gyro_data=gyro_data,
    sampling_freq_Hz=sampling_frequency,
    v_gyr_col_name="LowerBack_GYRO_x",
    ap_gyr_col_name="LowerBack_GYRO_z",
    ic_timestamps=initaal_contact_events[["onset"]],
    signal_type="vertical",
    recording=recording,
    tracking_system="LowerBack"
)

# Print results
print(f"Initial contacts and their corresponding labels:")
print(mccamley_ic_classifier.mccamley_df)
```

    Initial contacts and their corresponding labels:
            
            onset       duration       event_type           rl_label        tracking_system
    13      1348.700    0.0            initial contact      left            LowerBack
    14      1349.350    0.0            initial contact      right           LowerBack
    15      1349.975    0.0            initial contact      left            LowerBack
    16      1350.525    0.0            initial contact      right           LowerBack
    17      1351.050    0.0            initial contact      left            LowerBack
    ...     ...         ...            ...                  ...             ...
    2673    3475.750    0.0            initial contact      right           LowerBack
    2674    3476.300    0.0            initial contact      left            LowerBack
    2675    3476.825    0.0            initial contact      right           LowerBack
    2676    3477.375    0.0            initial contact      left            LowerBack
    2677    3477.925    0.0            initial contact      left            LowerBack

    [2665 rows x 5 columns]


### **Interpretation of Classified Initial Contacts**

Each row in the output represents a classified **initial contact** (IC) event, with the following columns:

- **`onset`** (`float`):  
  The timestamp (in seconds) when the initial contact occurred.

- **`duration`** (`float`):  
  Duration of the event. For initial contacts, this is typically set to `0.0`.

- **`event_type`** (`str`):  
  The type of event as `initial contact`.

- **`rl_label`** (`str`):  
  The classification of the initial contact as either `left` or `right`, based on the sign of the gyroscope signal at that timestamp.

- **`tracking_system`** (`str`):  
  The name of the tracking system (e.g., `"LowerBack"`) that was used to record the IMU data.