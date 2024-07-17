# Tutorial: Physical Activity Monitoring

**Author:** Masoud Abedinifar

**Last update:** Thu 14 Mar 2024

## Learning objectives  
By the end of this tutorial, you will be able to:

- Load accelerometer data from a recording that belongs to one of the available datasets, 
- Apply the Physical Activity Monitoring algorithm to classify activity intensity levels.  
- Visualize the activity intensity levels over time.
- Interpret the results of activity classification.  


## Physical Activity Monitoring

This example serves as a reference on how to use the physical activity monitoring algorithm. This example can be cited by referencing the package.

The example illustrates how the physical activity monitoring algorithm determines the intensity level of sedentary, light, moderate, and vigorous physical activities using body acceleration recorded with a triaxial accelerometer worn on the wrist. The physical activity monitoring algorithm is implemented in the main module [`kmat.modules.pam._pam`](https://github.com/neurogeriatricskiel/KMAT/tree/main/kmat/modules/pam/_pam.py).

The algorithm determines the intensity level of physical activities based on the following steps:

1. **Loading Data:** Start by loading the data, including a time index along with accelerometer data (N, 3) for x, y, and z axes. The other inputs are the sampling frequency of the data (sampling_freq_Hz), defaulting to 100 Hz, and thresholds (thresholds_mg), provided as a dictionary containing threshold values for physical activity detection in mg unit. Another input is the epoch duration (epoch_duration_sec) in seconds, defaulting to 5 seconds. The last input, plot_results, when set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each date, with a default of True.

2. **Preprocessing:** The input signal is preprocessed by calculating the sample-level Euclidean norm (EN) of the acceleration signal across the x, y, and z axes. A fourth-order Butterworth low-pass filter with a cut-off frequency of 20Hz is then applied to remove noise. This filter is applied to the vector magnitude scores. The ENMO index is calculated to separate the activity-related component of the acceleration signal. Negative ENMO values are truncated to zero. Finally, the indices are multiplied by 1000 to convert units from g to mg.

3. **Classification:** The algorithm classifies the intensity of physical activities based on the calculated ENMO values. The activity_classification function expresses the ENMO time-series data in 5-second epochs for summarizing the data. Thresholds for categorization are as follows: sedentary activity < 45 mg, light activity 45–100 mg, moderate activity 100–400 mg, vigorous activity > 400 mg.

4. **Results:** The algorithm classifies different levels of activities along with the time spent on each activity level for each day. If `plot_results` is set to True, the function generates a plot showing the averaged ENMO values for each day.

## References
[`1`] Doherty, Aiden, et al. (2017). Large scale population assessment of physical activity using wrist-worn accelerometers: the UK biobank study. PloS one 12.2. [https://doi.org/10.1371/journal.pone.0169649](https://doi.org/10.1371/journal.pone.0169649)

[`2`] Van Hees, Vincent T., et al. (2013). Separating movement and gravity components in an acceleration signal and implications for the assessment of human daily physical activity. PloS one 8.4. [https://doi.org/10.1371/journal.pone.0061691](https://doi.org/10.1371/journal.pone.0061691)


## Import Libraries
The necessary libraries such as pandas, physical activity monitoring and fairpark data loader are imported. Make sure that you have all the required libraries and modules installed before running this code. You may also need to install the 'kmat' library and its dependencies if you haven't already.



```python
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from kmat.modules.pam import PhysicalActivityMonitoring
from kmat.datasets import fairpark
from kmat.config import cfg_colors
```

## Data Preparation

To implement the physical activity monitoring algorithm, we load example data from a participant who has worn a wrist IMU sensor for several hours during a day while performing daily life activities at home.

The data, including a time index and accelerometer data (N, 3) for the x, y, and z axes, is loaded as a pandas DataFrame.


```python
# The 'file_path' variable holds the absolute path to the data file
file_path = r"C:\Users\Project\Desktop\sub_023\sub-023_imu-LARM_20160510_075144.csv"

# In this example, we use "imu" as tracking_system and "LARM" as tracked points.
tracking_sys = "imu"
tracked_points = {tracking_sys: ["LARM"]}

# The 'fairpark.load_recording' function is used to load the data
recording = fairpark.load_recording(
    file_name=file_path, tracking_systems=[tracking_sys], tracked_points=tracked_points
)

# Load and print wrist acceleration data (LARM)
acceleration_data = recording.data[tracking_sys][
    ["LARM_ACCEL_x", "LARM_ACCEL_y", "LARM_ACCEL_z"]
]
print(f"Acceleration data {recording.channels['imu']['units'][0]}:", acceleration_data)

# Get the corresponding sampling frequency from the recording
sampling_frequency = recording.channels["imu"]["sampling_frequency"].iloc[0]
```

    Acceleration data (m/s^2):     
                                LARM_ACCEL_x  LARM_ACCEL_y  LARM_ACCEL_z
    timestamp
    2016-10-05 07:51:44.000     -1.54         9.55          -2.11
    2016-10-05 07:51:44.010     -1.02         9.31          -1.94
    2016-10-05 07:51:44.020     -0.9          9.67          -1.64
    2016-10-05 07:51:44.030     -0.84         10.2          -2.17
    2016-10-05 07:51:44.040     -1.07         9.96          -2.11
    ...                         ...           ...           ...
    2016-10-05 18:31:49.590     -3.35         1.05          8.55
    2016-10-05 18:31:49.600     -2.88         0.25          9.26
    2016-10-05 18:31:49.610     -2.6          -0.25         9.71
    2016-10-05 18:31:49.620     -3.59         -0.84         10.24
    2016-10-05 18:31:49.630     -3.35         -1.13         10.59

[3840564 rows x 3 columns]


## Visualisation of the Data
The raw acceleration data including components of x, y and z axis is represented.

```python
# Create a figure
plt.figure(figsize=(14, 8))

# Get colors for raw (cfg_colors["raw"] contains color information)
colors = cfg_colors["raw"]  

# Plot acceleration data
for i in range(3):
    plt.plot(
        acceleration_data.index,
        acceleration_data[f"LARM_ACCEL_{chr(120 + i)}"],
        color=colors[i],
        label=f"Acc {'xyz'[i]}",
    )

# Add labels and legends to the figure
plt.xlabel("Time (h)", fontsize=16)
plt.ylabel("Acceleration (m/s$^2$)", fontsize=16)
plt.title("Acceleration Data from Wrist IMU Sensor")
plt.legend(fontsize=18)

# Formatting x-axis to display both time and date
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%Y-%m-%d"))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))

# Set custom x-axis ticks and limits
start_time = acceleration_data.index[0].replace(hour=0, minute=0, second=0)
end_time = start_time + pd.Timedelta(days=1)
xticks = pd.date_range(start_time, end_time, freq="H")

# Add a title
plt.title(
    "Accelerometer data from the wrist IMU sensor",
    fontsize=20,
)

# Customize tick and font sizes
plt.xticks(xticks, [f"{x.hour:02}" for x in xticks], fontsize=16)
plt.xlim(start_time, end_time)
plt.yticks(fontsize=16)

# Extract date from the first timestamp
date = str(acceleration_data.index[0].date())

# Add date underneath the x-axis labels
plt.text(0.5, -0.15, f"Date: {date}", fontsize=14, ha='center', transform=plt.gca().transAxes)

# Adjust legend location and its font
plt.legend(loc="upper left", fontsize=16)

# Display a grid
plt.grid(visible=None, which="both", axis="both")

# Show the plot
plt.tight_layout()
plt.show()
```

![](03_tutorial_physical_activity_monitoring_files/03_tutorial_physical_activity_monitoring_1.png)

## Apply Physical Activity Monitoring Algorithm
Now, we are running the physical activity monitoring algorithm from the main module [`kmat.modules.pam._pam`](https://github.com/neurogeriatricskiel/KMAT/tree/main/kmat/modules/pam/_pam.py). The inputs of the algorithm are as follows:

- **Input Data:** `data` Includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes in pandas Dataframe format.
- **Acceleration Unit:** `acceleration_unit` is the unit of the acceleration data.
- **Sampling Frequency:** `sampling_freq_Hz` is the sampling frequency of the acceleration data, defined in Hz, with a default value of 100 Hz.
- **Thresholds:** `thresholds_mg` are provided as a dictionary containing threshold values for physical activity detection in mili-g.
- **Epoch Duration:** `epoch_duration_sec` is the epoch length in seconds, with a default value of 5 seconds.
- **Plot Results:** `plot_results`, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each day. The default is True.

To apply the physical activity monitoring algorithm, an instance of the PhysicalActivityMonitoring class is created using the constructor, `PhysicalActivityMonitoring()`. The `pam` variable holds the instance, allowing us to access its methods. The output of the algorithm includes information regarding physical activity levels and the time spent on each activity for the provided date, including the mean of sedentary time, light, moderate, and vigorous activities, along with the time spent for each of them.

```python
# Create an instance of the PhysicalActivityMonitoring class
pam = PhysicalActivityMonitoring()

# Call phyisical activity monitoring using pam.detect
pam = pam.detect(
    data=acceleration_data,
    sampling_freq_Hz=sampling_frequency,
    thresholds_mg={
        "sedentary_threshold": 45,
        "light_threshold": 100,
        "moderate_threshold": 400,
    },
    epoch_duration_sec=5,
    plot=True,
)

# Phyisical activity information are stored in physical_activities_ attribute of pam
physical_activities = pam.physical_activities_

# Print daily phyisical activity information
print("Physical Activities Info:", physical_activities)
```
    Physical Activities Info:

        date            sedentary_mean_enmo   sedentary_time_min   light_mean_enmo    light_time_min    moderate_mean_enmo   moderate_time_min  vigorous_mean_enmo   vigorous_time_min
    0   2016-10-05      16.55                 600.16               52.13              39.66             111.67               0.33               NaN                  0.0


![](03_tutorial_physical_activity_monitoring_files/03_tutorial_physical_activity_monitoring_2.png)
    