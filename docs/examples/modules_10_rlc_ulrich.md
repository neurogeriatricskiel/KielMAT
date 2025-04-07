# Tutorial: Ullrich Initial Contact Classification

**Author:** Masoud Abedinifar 

**Last update:** Mon 07 April 2025

## Learning Objectives
By the end of this tutorial, you will be able to:

- Load data from [`Mobilised`](https://github.com/neurogeriatricskiel/KielMAT/blob/main/kielmat/datasets/mobilised.py), one of the datasets provided in the KielMAT toolbox.
- Apply the [`Ullrich Initial Contact Classification`](https://github.com/neurogeriatricskiel/KielMAT/blob/main/kielmat/modules/rlc/_ulrich.py) algorithm to assign left or right labels to detected initial contacts using machine learning.
- Analyze the output and integrate the classified initial contacts into a `KielMATRecording` object for further gait analysis.

## Ullrich Initial Contact Classification

This example can be referenced by citing the package and the original research paper by Ullrich et al. [1].

The example demonstrates the application of the Ullrich Initial Contact Classification algorithm for distinguishing left and right initial contacts using gyroscope data collected from a lower-back IMU sensor. The classification algorithm is implemented using [`kielmat.modules.rlc._ulrich`](https://github.com/neurogeriatricskiel/KielMAT/blob/main/kielmat/modules/rlc/_ulrich.py). This machine learning-based method builds upon earlier heuristic approaches, such as McCamley et al. [2].

Unlike signal sign-based classification, the Ullrich algorithm extracts a 6-dimensional feature vector from the vertical and anterior-posterior gyroscope components at each detected initial contact (IC). These features include:

- Filtered vertical and anterior-posterior gyroscope signals
- First derivatives of both signals
- Second derivatives of both signals

The signals are first preprocessed using a Butterworth bandpass filter to enhance signal quality and reduce noise. After feature extraction, the pre-trained machine learning model is used to classify each IC as either `left` or `right`.

The method supports multiple model types:
- **Random Forest Classifier** (`rfc`)
- **Support Vector Machine** (`svm linear or RBF kernel`)
- **K-Nearest Neighbors** (`knn`)

The classification results are stored in a pandas DataFrame (`ulrich_df` attribute), containing:
- `onset`: Time of initial contact (in seconds)
- `duration`: Set to 0.0
- `event_type`: Always `initial contact`
- `rl_label`: Predicted laterality (`left` or `right`)
- `tracking_system`: Label for the sensor setup (e.g., "SU")

If a `KielMATRecording` object is provided, the classification results are automatically integrated into its event structure.

**References**

[1] **Ullrich et al.** (2022). *Machine Learning-Based Distinction of Left and Right Foot Contacts in Lower Back Inertial Sensor Data Improves Gait Analysis Accuracy.* IEEE Transactions on Neural Systems and Rehabilitation Engineering.  
[https://doi.org/10.1109/EMBC46164.2021.9630653](https://doi.org/10.1109/EMBC46164.2021.9630653)

[2] **McCamley et al.** (2012). *An Enhanced Estimate of Initial Contact and Final Contact Instants of Time Using Lower Trunk Inertial Sensor Data.* Gait & Posture, 36(2), 318–320.  
[https://doi.org/10.1016/j.gaitpost.2012.02.019](https://doi.org/10.1016/j.gaitpost.2012.02.019)


## Import Libraries

In this section, we import the required Python libraries such as `numpy`, `pandas`, and `matplotlib.pyplot` along with the relevant modules from the `kielmat` package. Most importantly, we import the **Ullrich Initial Contact Classification** algorithm from `kielmat.modules.rlc`.

Make sure you have the `kielmat` toolbox and its dependencies installed in your environment. This includes support for working with the **Mobilised** dataset, signal preprocessing, and machine learning-based classification.

If you haven't installed the package yet, refer to the [KielMAT GitHub repository](https://github.com/neurogeriatricskiel/KielMAT) for installation instructions.

```python
import numpy as np
import pandas as pd
import os
from kielmat.datasets import mobilised
from kielmat.modules.gsd import ParaschivIonescuGaitSequenceDetection
from kielmat.modules.icd import ParaschivIonescuInitialContactDetection
from kielmat.modules.rlc import UllrichInitialContactClassification
from pathlib import Path
```

## Data Preparation

To implement the MacCamley Initial Contact Classification algortihm, we load example data from publicly available on the Zenodo repository [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7547125.svg)](https://doi.org/10.5281/zenodo.7547125). 

The participant was monitored for 2.5 hours in a real-world setting while performing various daily activities. Additionally, structured tasks such as outdoor walking, stair ascent/descent, and transitions between rooms were included in the assessment [`3`].

**References**

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

## Applying Ullrich Initial Contact Classification Algorithm

In this step, we apply the **Ullrich Initial Contact Classification** algorithm from the [`kielmat.modules.rlc._ulrich.UllrichInitialContactClassification`](https://github.com/neurogeriatricskiel/KielMAT/blob/main/kielmat/modules/rlc/_ulrich.py) module to determine the laterality (left/right) of initial contacts detected in gait sequences.

Prior to classification, we use the **Paraschiv-Ionescu Gait Sequence Detection** algorithm to identify gait sequences based on lower-back acceleration data. This is achieved using the [`kielmat.modules.gsd._paraschiv.ParaschivIonescuGaitSequenceDetection`](https://github.com/neurogeriatricskiel/KielMAT/blob/main/kielmat/modules/gsd/_paraschiv.py) module.

Once gait sequences are detected, we apply the **Paraschiv-Ionescu Initial Contact Detection** algorithm to detect precise initial contact (IC) events. These ICs then serve as the input for the Ullrich classifier, which uses gyroscope-based features to label each IC as `left` or `right` using a pre-trained machine learning model.

```python
# Create an instance of the ParaschivIonescuGaitSequenceDetection class
gsd = ParaschivIonescuGaitSequenceDetection()

# Call the gait sequence detection using gsd.detect to detect gait sequences
gsd = gsd.detect(
    accel_data=accel_data, 
    sampling_freq_Hz=sampling_frequency, 
    tracking_system="SU", 
    plot_results=False
)

# Add events to the recording as a dictionary including tracking system and events
recording.add_events(tracking_system="SU", new_events=gsd.gait_sequences_)

# Filter only gait sequence events
gait_sequence_events = recording.events["SU"][recording.events["SU"]["event_type"] == "gait sequence"]

# Print the filtered gait sequences
print(f'Gait sequences and thier corresponding information:')
print(gait_sequence_events)
```

        13 gait sequence(s) detected.
        Gait sequences and thier corresponding information:
                    onset       duration        event_type          tracking_system
            0       1348.200    7.050           gait sequence       SU
            1       1372.025    4.600           gait sequence       SU
            2       1388.750    199.675         gait sequence       SU
            3       1596.050    24.700          gait sequence       SU
            4       1694.750    457.475         gait sequence       SU
            5       2168.000    310.625         gait sequence       SU
            6       2585.900    43.700          gait sequence       SU
            7       2633.625    315.600         gait sequence       SU
            8       2956.600    9.475           gait sequence       SU
            9       3029.725    16.450          gait sequence       SU
            10      3087.550    9.825           gait sequence       SU
            11      3195.125    12.825          gait sequence       SU
            12      3464.250    14.375          gait sequence       SU


Next, we apply the **Paraschiv-Ionescu Initial Contact Detection** algorithm to detect initial contacts within these gait sequences using [`KielMAT.kielmat.modules.icd._paraschiv.ParaschivIonescuInitialContactDetection`](https://github.com/neurogeriatricskiel/KielMAT/tree/main/kielmat/modules/icd/_paraschiv.py). 


```python
# Now, use Paraschiv-Ionescu initial contact detection algortihm to find initial contacts within detected gait sequences.
icd = ParaschivIonescuInitialContactDetection()

# Call the initial contact detection using icd.detect
icd = icd.detect(
    accel_data=accel_data,
    gait_sequences=gsd.gait_sequences_,
    sampling_freq_Hz=sampling_frequency,
    tracking_system="SU", 
    v_acc_col_name="LowerBack_ACCEL_x"
)

# Add events to the recording as a dictionary including tracking system and events
recording.add_events(tracking_system="SU", new_events=icd.initial_contacts_)

# Filter only gait sequence events
initaal_contact_events = recording.events["SU"][recording.events["SU"]["event_type"] == "initial contact"]

# Print the filtered gait sequences
print(f"Initial contacts information:")
print(initaal_contact_events)
```

        Initial contacts information:
                onset       event_type          duration    tracking_systems

        0       1348.700    initial contact     0           SU
        1       1349.350    initial contact     0           SU
        2       1349.975    initial contact     0           SU
        3       1350.525    initial contact     0           SU
        4       1351.050    initial contact     0           SU
        ...     ...         ...                 ...         ...
        2660    3475.750    initial contact     0           SU
        2661    3476.300    initial contact     0           SU
        2662    3476.825    initial contact     0           SU
        2663    3477.375    initial contact     0           SU
        2664    3477.925    initial contact     0           SU

        [2665 rows x 4 columns]


Once the initial contacts are detected, the **Ullrich Initial Contact Classification** algorithm is used to classify them as `left` or `right` using machine learning models trained on gyroscope-derived features.

Inputs for Ullrich Initial Contact Classification Algorithm:

- **`gyro_data`** (`pd.DataFrame`):  
  A DataFrame containing gyroscope signals with shape (N, 3), typically including x, y, and z components from a lower-back IMU.

- **`sampling_freq_Hz`** (`float`):  
  Sampling frequency of the gyroscope signals in Hertz (Hz).

- **`v_gyr_col_name`** (`str`):  
  Name of the column in `gyro_data` that corresponds to the **vertical gyroscope component** (usually aligned with the x-axis of the sensor).

- **`ap_gyr_col_name`** (`str`):  
  Name of the column in `gyro_data` that corresponds to the **anterior-posterior gyroscope component** (typically z-axis).

- **`ic_timestamps`** (`pd.DataFrame`):  
  A DataFrame containing detected initial contact (IC) timestamps. Must include an `onset` column with timestamps in seconds.

- **`ml_model_type`** (`str`):  
  Specifies the type of machine learning model to use for classification. Options include:
  
  - `"rfc"`: Random Forest Classifier  
  - `"svm_linear"`: Support Vector Machine with linear kernel  
  - `"svm_rbf"`: Support Vector Machine with RBF kernel  
  - `"knn"`: K-Nearest Neighbors classifier

- **`recording`** (`KielMATRecording`, optional):  
  If provided, the output labels (`rl_label`) will be directly added to the corresponding initial contact rows in the `recording.events` table.

- **`tracking_system`** (`str`, required if `recording` is provided):  
  The tracking system key (e.g., `"SU"`) that identifies which part of the recording's event table to update.


```python
# Create an instance of the classifier
ulrich_classifier = UllrichInitialContactClassification()

# Apply the classifier
ulrich_classifier = ulrich_classifier.detect(
    gyro_data=gyro_data,
    sampling_freq_Hz=sampling_frequency,
    v_gyr_col_name="LowerBack_GYRO_x",      # vertical
    ap_gyr_col_name="LowerBack_GYRO_z",     # anterior-posterior
    ic_timestamps=initaal_contact_events,   # detected ICs from Paraschiv-Ionescu
    ml_model_type="rfc",                    # choose between 'rfc', 'svm_linear', 'svm_rbf' or 'knn'
    recording=recording,
    tracking_system="SU"
)

# Print results
print(f"Initial contacts and their corresponding labels:")
print(ulrich_classifier.ulrich_df)
```

    Initial contacts and their corresponding labels:
            
            onset       duration       event_type           rl_label        tracking_system
    13      1348.700    0.0            initial contact      left            SU
    14      1349.350    0.0            initial contact      right           SU
    15      1349.975    0.0            initial contact      left            SU
    16      1350.525    0.0            initial contact      right           SU
    17      1351.050    0.0            initial contact      left            SU
    ...     ...         ...            ...                  ...             ...
    2673    3475.750    0.0            initial contact      right           SU
    2674    3476.300    0.0            initial contact      left            SU
    2675    3476.825    0.0            initial contact      right           SU
    2676    3477.375    0.0            initial contact      left            SU
    2677    3477.925    0.0            initial contact      left            SU

    [2665 rows x 5 columns]