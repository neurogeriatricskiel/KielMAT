# Tutorial: Physical Activity Monitoring

**Author:** Masoud Abedinifar

**Last update:** Fri 09 August 2024

## Learning objectives  
By the end of this tutorial, you will be able to:  

- Load accelerometer data from a raw recording
- Apply the Physical Activity Monitoring algorithm to classify activity intensity levels.  
- Interpret the results of activity classification.  

# Physical Activity Monitoring

This example serves as a reference on how to use the physical activity monitoring algorithm. This example can be cited by referencing the package.

The example illustrates how the physical activity monitoring algorithm determines the intensity level of sedentary, light, moderate, and vigorous physical activities using body acceleration recorded with a triaxial accelerometer worn on the lowerback. The physical activity monitoring algorithm is implemented in the main module [`kielmat.modules.pam._pam`](https://github.com/neurogeriatricskiel/KielMAT/tree/main/kielmat/modules/pam/_pam.py).

The algorithm determines the intensity level of physical activities based on the following steps:

1. **Loading Data:** Start by loading the data, including a time index along with accelerometer data (N, 3) for x, y, and z axes. The other inputs are the sampling frequency of the data (sampling_freq_Hz), defaulting to 100 Hz, and thresholds (thresholds_mg), provided as a dictionary containing threshold values for physical activity detection in mg unit. Another input is the epoch duration (epoch_duration_sec) in seconds, defaulting to 5 seconds. The last input, plot_results, when set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each date, with a default of True.

2. **Preprocessing:** The input signal is preprocessed by calculating the sample-level Euclidean norm (EN) of the acceleration signal across the x, y, and z axes. A fourth-order Butterworth low-pass filter with a cut-off frequency of 20Hz is then applied to remove noise. This filter is applied to the vector magnitude scores. The ENMO index is calculated to separate the activity-related component of the acceleration signal. Negative ENMO values are truncated to zero. Finally, the indices are multiplied by 1000 to convert units from g to mg.

3. **Classification:** The algorithm classifies the intensity of physical activities based on the calculated ENMO values. The activity_classification function expresses the ENMO time-series data in 5-second epochs for summarizing the data. Thresholds for categorization are as follows: sedentary activity < 45 mg, light activity 45–100 mg, moderate activity 100–400 mg, vigorous activity > 400 mg.

4. **Results:** The algorithm classifies different levels of activities along with the time spent on each activity level for each day. If `plot_results` is set to True, the function generates a plot showing the averaged ENMO values for each day.

#### References
[`1`] Doherty, Aiden, et al. (2017). Large scale population assessment of physical activity using wrist-worn accelerometers: the UK biobank study. PloS one 12.2. [https://doi.org/10.1371/journal.pone.0169649](https://doi.org/10.1371/journal.pone.0169649)

[`2`] Van Hees, Vincent T., et al. (2013). Separating movement and gravity components in an acceleration signal and implications for the assessment of human daily physical activity. PloS one 8.4. [https://doi.org/10.1371/journal.pone.0061691](https://doi.org/10.1371/journal.pone.0061691)

## Import Libraries
The necessary libraries such as pandas, physical activity monitoring and mobilised dataset are imported. Make sure that you have all the required libraries and modules installed before running this code. You may also need to install the `kielmat` library and its dependencies if you haven't already.


```python
import pandas as pd
import os
from pathlib import Path
from kielmat.modules.pam import PhysicalActivityMonitoring
from kielmat.datasets import mobilised
```

## Data Preparation

To implement the physical activity monitoring algorithm, we load example data from a participant who has worn a LowerBack IMU sensor for several hours during a day while performing daily life activities at home.

The accelerometer data (N, 3) for the x, y, and z axes, is loaded as a pandas DataFrame.

```python
# Set the dataset path
dataset_path = Path(os.getcwd()) / "_mobilised"

# Fetch and load the dataset
mobilised.fetch_dataset(dataset_path=dataset_path)
```

```python
# In this example, we use "SU" as tracking_system and "LowerBack" as tracked points.
tracking_sys = "SU"
tracked_points = {tracking_sys: ["LowerBack"]}
```


```python
# The 'mobilised.load_recording' function is used to load the data from the specified file_path
recording = mobilised.load_recording(
    cohort="PFF",  # Choose the cohort
    file_name="data.mat", 
    dataset_path=dataset_path)

# Load lower back acceleration data
accel_data = recording.data[tracking_sys][
    ["LowerBack_ACCEL_x", "LowerBack_ACCEL_y", "LowerBack_ACCEL_z"]
]

# Get the corresponding sampling frequency directly from the recording
sampling_frequency = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"] == "LowerBack_ACCEL_x"
]["sampling_frequency"].values[0]

# Get the acceleration data unit from the recording
acceleration_unit = recording.channels[tracking_sys][
    recording.channels[tracking_sys]["name"] == "LowerBack_ACCEL_x"
]["units"].values[0]
```

## Apply Physical Activity Monitoring Algorithm
Now, we are running the physical activity monitoring algorithm from the main module [`kielmat.modules.pam._pam`](https://github.com/neurogeriatricskiel/KielMAT/tree/main/kielmat/modules/pam/_pam.py). The inputs of the algorithm are as follows:

- **Input Data:** `data` Includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes in pandas Dataframe format.
- **Acceleration Unit:** `acceleration_unit` is the unit of the acceleration data.
- **Sampling Frequency:** `sampling_freq_Hz` is the sampling frequency of the acceleration data, defined in Hz, with a default value of 100 Hz.
- **Thresholds:** `thresholds_mg` are provided as a dictionary containing threshold values for physical activity detection in mili-g.
- **Epoch Duration:** `epoch_duration_sec` is the epoch length in seconds, with a default value of 5 seconds.
- **Plot Results:** `plot_results`, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each day.

To apply the physical activity monitoring algorithm, an instance of the PhysicalActivityMonitoring class is created using the constructor, `PhysicalActivityMonitoring()`. The `pam` variable holds the instance, allowing us to access its methods. The output of the algorithm includes information regarding physical activity levels and the time spent on each activity for the provided date, including the mean of sedentary time, light, moderate, and vigorous activities, along with the time spent for each of them.


```python
# Initialize the PhysicalActivityMonitoring class
pam = PhysicalActivityMonitoring()

# Detect physical activity
pam.detect(
    data=accel_data,
    acceleration_unit=acceleration_unit,
    sampling_freq_Hz=sampling_frequency,
    thresholds_mg={
        "sedentary_threshold": 45,
        "light_threshold": 100,
        "moderate_threshold": 400,
    },
    epoch_duration_sec=5,
    plot=False
)

# Print detected physical activities
print(pam.physical_activities_)
```
        date        sedentary_mean_enmo  sedentary_time_min  light_mean_enmo     light_time_min  moderate_mean_enmo  moderate_time_min      vigorous_time_min  vigorous_mean_enmo  
    0  2023-01-01   0.824444             115.583333          NaN                 0               NaN                 0                      NaN                0           





