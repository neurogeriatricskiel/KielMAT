# Gait Sequence Detection

This example illustrates how the gait sequence detection (GSDB) algorithm
can be used to detect gait sequences using body acceleration recorded with a triaxial accelerometer worn/fixed on the lower back. The gait sequence detection algorithm is implemented in the main module GSDB.py. 

The algorithm detects the gait sequences based on identified steps. The algorithm starts with loading the accelerometer data including three columns corresponding to the acceleration signal across the x, y, and z axes, as well as the sampling frequency of the data. To simplify analysis, the norm of acceleration across the x, y, and z axes is computed. Then, the signal is resampled at 40 Hz sampling frequency using interpolation. Then, smoothing is applied through a Savitzky-Golay filter and a Finite Impulse Response (FIR) lowpass filter to remove noise and drifts from the signal. Continuous wavelet transform is then applied to capture gait-related features, followed by additional smoothing using successive Gaussian-weighted filters. The processed data is then analyzed to detect gait sequences.

Then, the code continues by identifying the envelope of the processed acceleration signal. The active periods of the signal are identified using the Hilbert envelope. The statistical distribution of the amplitude of the peaks in these active periods is then used to derive an adaptive threshold. In case the Hilbert envelope algorithm fails to detect active periods, the fixed threshold value (0.15 g) is used for peak detection in the signal. Mid-swing peaks are detected based on this threshold. Pulse trains in the local maximum and minimum of the peaks are identified, with those having fewer than four steps filtered out. The intersection of pulse trains from local maximum and minimum peaks is detected as walking periods. These periods are then organized and packed to update the start and end times of detected walking bouts.

Then, final steps are taken to detect walking bouts in the signal. For this purpose, walking bouts with five or more steps are detected, and their start and end times are added to the list. Walking labels are generated as an array of zeros, and the intervals corresponding to the walking bouts are labeled as 1. Then, groups of consecutive zeros in the walking labels are identified, and if breaks between walking bouts are less than three seconds, they are merged. Finally, the output is constructed as a list of dictionaries containing the start and end times of detected gait sequences. If gait sequences are found, the output is printed; otherwise, a message indicating that no gait sequences are detected is displayed. Optionally, if the plot_results flag is set to True, a visualization plot is generated to display the preprocessed data and detected gait sequences. The pipeline also comprises several error-checking methods to maintain data integrity. 

## Import libraries
The necessary libraries are imported. 


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from ngmt.utils import matlab_loader
from ngmt.modules.gsd import GSDB
```

## Getting example data
The algorithm was developed and validated using the data of the following cohorts: congestive heart failure (CHF), chronic obstructive pulmonary disease (COPD), healthy adult (HA), multiple sclerosis (MS), Parkinson’s (PD) and proximal femoral facture (PFF) included in the Zenodo repository ("https://ADDRESS_SHOULD_BE_ADDED_HERE").

To implemet the GSDB algortihm, the example data from CHF cohort is loaded.


```python
# Data directory
file_directory = 'C:\\Users\\Project\\Desktop\\Gait_Sequence\\Mobilise-D dataset_1-18-2023\\CHF\\data.mat'

# Load IMU sendor data using load_matlab 
data_dict = matlab_loader.load_matlab(file_directory, top_level="data")

# Get the acceleration data
acceleration_data = data_dict['TimeMeasure1']['Recording4']['SU']["LowerBack"]["Acc"]

# Get the corresponding sampling frequency
sampling_frequency = data_dict['TimeMeasure1']['Recording4']['SU']["LowerBack"]['Fs']["Acc"]

```

## Applying the gait sequence detection algorithm
First we need to initialize gait sequence detection (GSDB) algorithm.
In most cases it is sufficient to keep all parameters at default.


```python
# Use Gait_Sequence_Detection to detect gait sequence 
gait_sequences = GSDB.Gait_Sequence_Detection(imu_acceleration=acceleration_data, sampling_frequency=sampling_frequency, plot_results=True)

# Display the detected gait sequences
if gait_sequences:
    for i, sequence in enumerate(gait_sequences):
        print(f"Gait Sequence {i + 1}:")
        print(f"Start Time: {sequence['Start']} seconds")
        print(f"End Time: {sequence['End']} seconds")
        print(f"Sampling Frequency: {sequence['fs']} Hz")
        print()
```
