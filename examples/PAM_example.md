# PHysical Activity Monitoring (PAM)

This notebook will serve as a reference on how to use phyisical activity monitoring algortihm.

The example illustrates how the physical activity monitoring (PAM) algorithm is used to determine the intensity level of sedentary, light, moderate, and vigorous physical activities using body acceleration recorded with a triaxial accelerometer worn on the wrist. The physical activity monitoring algorithm is implemented in the main module `NGMT\ngmt\modules\PAM.py`.

The algorithm determines the intensity level of the physical activities based on the following steps. It starts by loading the input_data, which includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes. The other input of the algorithm is sampling_frequency, which is in Hz, and the default value is 100. Another input of the algorithm is thresholds, which should be provided as a dictionary containing threshold values for physical activity detection. Epoch length, which is defined in seconds, is the next input of the algorithm, as shown with epoch_duration. The default value is 5 seconds. The last input of the algorithm is plot_results, which, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each date. The default is True.

The following steps are taken to preprocess the input signal. First, the sample-level Euclidean norm (EN) of the acceleration signal across the x, y, and z axes is calculated. Next, a fourth-order Butterworth low-pass filter with a cut-off frequency of 20Hz is applied to remove noise. This filter is applied to the vector magnitude scores, rather than the individual axes. To separate out the activity-related component of the acceleration signal, the ENMO index is calculated. The Euclidean Norm Minus One (ENMO) is a summary metric for acceleration data and represents the vector magnitude of 3 axial measures minus the contribution of gravity (1 g). Then, negative values of the ENMO are truncated to zero. Finally, the calculated indices are multiplied by 1000 to convert the units of the acceleration from g to milli-g.

The algorithm continues by classifying the intensity of the physical activities based on the calculated ENMO values. Using the activity_classification function, the ENMO time-series data is then expressed in 5-second epochs. Epochs with a length of 5 seconds are used for summarizing the data, as this epoch length has been suggested to be able to capture shorter bouts of activities. The greater the intensity of movement and duration of activity in the summed 5-second epochs are, the greater the ENMO value is. Then, the intensity of activities as the time distribution of ENMO using 5-second epochs is used to classify activities based on different thresholds. In the analysis of intensity distribution, the following thresholds are used for categorization: sedentary activity < 45 milli-g, light activity 45–100 milli-g, moderate activity 100–400 milli-g, vigorous activity > 400 milli-g.

Finally, the algorithm takes the last steps to classify different levels of activities along with the time spent on each activity level for each day. The algorithm also visualizes the averaged ENMO values for each day.


## Import libraries
The necessary libraries such as pandas, os and physical activity monitoring (PAM) are imported. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the 'ngmt' library and its dependencies if you haven't already.


```python
import pandas as pd
import numpy as np
from ngmt.modules import PAM
import os
```

## Load Data
The data, including time index and accelerometer data (N, 3) for the x, y, and z axes, is loaded as a numpy.ndarray.


```python
# This code snippet loads motion data from a csv file.
# The data importer for fair-park data will be added.
# The 'data_folder_path' variable holds the absolute path to the data file
data_folder_path = r"C:\Users\Project\Desktop\sub_001"

# Create an empty DataFrame to store the combined data
combined_data = pd.DataFrame()

# Iterate over all CSV files in the folder for each subject
for file_name in os.listdir(data_folder_path):
    # Check if the file is a CSV file and has the expected prefix
    if file_name.endswith(".csv") and file_name.startswith("sub-001_imu-LARM_"):
        # Construct the full path to the CSV file
        file_path = os.path.join(data_folder_path, file_name)

        # Read the CSV file into a DataFrame
        current_data = pd.read_csv(file_path, header=None, sep=";")

        # Rename columns and convert time columns to datetime format
        current_data.columns = [
            "Acc_x",
            "Acc_y",
            "Acc_z",
            "Gyr_x",
            "Gyr_y",
            "Gyr_z",
            "Mag_x",
            "Mag_y",
            "Mag_z",
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Second",
        ]
        current_data["time"] = pd.to_datetime(
            current_data[["Year", "Month", "Day", "Hour", "Minute", "Second"]],
            format="%Y-%m-%d %H:%M:%S",
        )
        current_data.set_index("time", inplace=True)

        # Select accelerometer columns and convert units from m/s^2 to g
        current_data = current_data[["Acc_x", "Acc_y", "Acc_z"]].copy()
        current_data[["Acc_x", "Acc_y", "Acc_z"]] /= 9.81

        # Concatenate the current data with the combined data
        combined_data = pd.concat([combined_data, current_data])

# Sampling frequency
sampling_frequency = 100
```

## Apply the physical activity monitoring algorithm
Now, we are running physical activity monitoring (PAM) algorithm from main module (`NGMT.ngmt.modules.PAM.Physical_Activity_Monitoring`). The inputs of the algorihm are as follows. The input_data, includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes. The sampling_frequency is the sampling frequecy of the acceleration data which is defined in Hz, and the default value of it is 100 Hz. The next input, thresholds, are provided as a dictionary containing threshold values for physical activity detection. The epoch_duration is the epoch length in seconds and the default value is 5 seconds. The last input of the algorithm is plot_results, which, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each day. The default is True.





```python
# Use Physical_Activity_Monitoring to classify physical activity levels
phyam_results = PAM.Physical_Activity_Monitoring(
    input_data=combined_data,
    sampling_frequency=100,
    thresholds={
        "sedentary_threshold": 45,
        "light_threshold": 100,
        "moderate_threshold": 400,
    },
    epoch_duration=5,
    plot_results=True,
)
```


    
![png](PAM_example_files/PAM_example_6_0.png)
    



```python
# Display average ENMO values for each activity level for each day.
for index, row in phyam_results.iterrows():
    print(f"Date: {row['date']}")

    # Check if there are NaN values for mean acceleration
    if not np.isnan(row["sedentary_mean_acc"]):
        print(f"Mean ENMO for Sedentary: {row['sedentary_mean_acc']} mili-g")
        print(f"Time Spent for Sedentary: {row['sedentary_spent_time_minute']} minutes")

    if not np.isnan(row["light_mean_acc"]):
        print(f"Mean ENMO for Light Activity: {row['light_mean_acc']} mili-g")
        print(
            f"Time Spent for Light Activity: {row['light_spent_time_minute']} minutes"
        )

    if not np.isnan(row["moderate_mean_acc"]):
        print(f"Mean ENMO for Moderate Activity: {row['moderate_mean_acc']} mili-g")
        print(
            f"Time Spent for Moderate Activity: {row['moderate_spent_time_minute']} minutes"
        )

    if not np.isnan(row["vigorous_mean_acc"]):
        print(f"Mean ENMO for Vigorous Activity: {row['vigorous_mean_acc']} mili-g")
        print(
            f"Time Spent for Vigorous Activity: {row['vigorous_spent_time_minute']} minutes"
        )

    print("=" * 40)
```

    Date: 2016-10-21
    Mean ENMO for Sedentary: 5.315398506372166 mili-g
    Time Spent for Sedentary: 338.41666666666663 minutes
    Mean ENMO for Light Activity: 69.80800287441187 mili-g
    Time Spent for Light Activity: 145.16666666666666 minutes
    Mean ENMO for Moderate Activity: 119.97283014741834 mili-g
    Time Spent for Moderate Activity: 15.083333333333332 minutes
    ========================================
    Date: 2016-10-22
    ========================================
    Date: 2016-10-23
    ========================================
    Date: 2016-10-24
    ========================================
    Date: 2016-10-25
    ========================================
    Date: 2016-10-26
    ========================================
    Date: 2016-10-27
    ========================================
    Date: 2016-10-28
    ========================================
    Date: 2016-10-29
    ========================================
    Date: 2016-10-30
    ========================================
    Date: 2016-10-31
    ========================================
    Date: 2016-11-01
    ========================================
    Date: 2016-11-02
    ========================================
    Date: 2016-11-03
    ========================================
    Date: 2016-11-04
    ========================================
    Date: 2016-11-05
    ========================================
    Date: 2016-11-06
    ========================================
    Date: 2016-11-07
    ========================================
    Date: 2016-11-08
    ========================================
    Date: 2016-11-09
    ========================================
    Date: 2016-11-10
    ========================================
    Date: 2016-11-11
    ========================================
    Date: 2016-11-12
    ========================================
    Date: 2016-11-13
    ========================================
    Date: 2016-11-14
    ========================================
    Date: 2016-11-15
    ========================================
    Date: 2016-11-16
    ========================================
    Date: 2016-11-17
    ========================================
    Date: 2016-11-18
    ========================================
    Date: 2016-11-19
    ========================================
    Date: 2016-11-20
    ========================================
    Date: 2016-11-21
    ========================================
    Date: 2016-11-22
    ========================================
    Date: 2016-11-23
    ========================================
    Date: 2016-11-24
    ========================================
    Date: 2016-11-25
    ========================================
    Date: 2016-11-26
    ========================================
    Date: 2016-11-27
    ========================================
    Date: 2016-11-28
    ========================================
    Date: 2016-11-29
    ========================================
    Date: 2016-11-30
    ========================================
    Date: 2016-12-01
    ========================================
    Date: 2016-12-02
    ========================================
    Date: 2016-12-03
    ========================================
    Date: 2016-12-04
    ========================================
    Date: 2016-12-05
    ========================================
    Date: 2016-12-06
    ========================================
    Date: 2016-12-07
    ========================================
    Date: 2016-12-08
    ========================================
    Date: 2016-12-09
    ========================================
    Date: 2016-12-10
    ========================================
    Date: 2016-12-11
    ========================================
    Date: 2016-12-12
    ========================================
    Date: 2016-12-13
    ========================================
    Date: 2016-12-14
    ========================================
    Date: 2016-12-15
    ========================================
    Date: 2016-12-16
    ========================================
    Date: 2016-12-17
    ========================================
    Date: 2016-12-18
    ========================================
    Date: 2016-12-19
    ========================================
    Date: 2016-12-20
    ========================================
    Date: 2016-12-21
    ========================================
    Date: 2016-12-22
    ========================================
    Date: 2016-12-23
    ========================================
    Date: 2016-12-24
    ========================================
    Date: 2016-12-25
    ========================================
    Date: 2016-12-26
    ========================================
    Date: 2016-12-27
    ========================================
    Date: 2016-12-28
    ========================================
    Date: 2016-12-29
    ========================================
    Date: 2016-12-30
    ========================================
    Date: 2016-12-31
    ========================================
    Date: 2017-01-01
    ========================================
    Date: 2017-01-02
    ========================================
    Date: 2017-01-03
    ========================================
    Date: 2017-01-04
    ========================================
    Date: 2017-01-05
    ========================================
    Date: 2017-01-06
    ========================================
    Date: 2017-01-07
    ========================================
    Date: 2017-01-08
    ========================================
    Date: 2017-01-09
    ========================================
    Date: 2017-01-10
    ========================================
    Date: 2017-01-11
    ========================================
    Date: 2017-01-12
    ========================================
    Date: 2017-01-13
    ========================================
    Date: 2017-01-14
    ========================================
    Date: 2017-01-15
    ========================================
    Date: 2017-01-16
    ========================================
    Date: 2017-01-17
    ========================================
    Date: 2017-01-18
    ========================================
    Date: 2017-01-19
    ========================================
    Date: 2017-01-20
    ========================================
    Date: 2017-01-21
    ========================================
    Date: 2017-01-22
    ========================================
    Date: 2017-01-23
    ========================================
    Date: 2017-01-24
    ========================================
    Date: 2017-01-25
    ========================================
    Date: 2017-01-26
    ========================================
    Date: 2017-01-27
    ========================================
    Date: 2017-01-28
    ========================================
    Date: 2017-01-29
    ========================================
    Date: 2017-01-30
    ========================================
    Date: 2017-01-31
    ========================================
    Date: 2017-02-01
    ========================================
    Date: 2017-02-02
    ========================================
    Date: 2017-02-03
    ========================================
    Date: 2017-02-04
    ========================================
    Date: 2017-02-05
    ========================================
    Date: 2017-02-06
    ========================================
    Date: 2017-02-07
    ========================================
    Date: 2017-02-08
    ========================================
    Date: 2017-02-09
    ========================================
    Date: 2017-02-10
    ========================================
    Date: 2017-02-11
    ========================================
    Date: 2017-02-12
    ========================================
    Date: 2017-02-13
    ========================================
    Date: 2017-02-14
    ========================================
    Date: 2017-02-15
    ========================================
    Date: 2017-02-16
    ========================================
    Date: 2017-02-17
    ========================================
    Date: 2017-02-18
    ========================================
    Date: 2017-02-19
    ========================================
    Date: 2017-02-20
    ========================================
    Date: 2017-02-21
    ========================================
    Date: 2017-02-22
    ========================================
    Date: 2017-02-23
    ========================================
    Date: 2017-02-24
    ========================================
    Date: 2017-02-25
    ========================================
    Date: 2017-02-26
    ========================================
    Date: 2017-02-27
    ========================================
    Date: 2017-02-28
    ========================================
    Date: 2017-03-01
    ========================================
    Date: 2017-03-02
    ========================================
    Date: 2017-03-03
    ========================================
    Date: 2017-03-04
    ========================================
    Date: 2017-03-05
    ========================================
    Date: 2017-03-06
    ========================================
    Date: 2017-03-07
    ========================================
    Date: 2017-03-08
    ========================================
    Date: 2017-03-09
    ========================================
    Date: 2017-03-10
    ========================================
    Date: 2017-03-11
    ========================================
    Date: 2017-03-12
    ========================================
    Date: 2017-03-13
    ========================================
    Date: 2017-03-14
    ========================================
    Date: 2017-03-15
    ========================================
    Date: 2017-03-16
    ========================================
    Date: 2017-03-17
    ========================================
    Date: 2017-03-18
    ========================================
    Date: 2017-03-19
    ========================================
    Date: 2017-03-20
    ========================================
    Date: 2017-03-21
    ========================================
    Date: 2017-03-22
    ========================================
    Date: 2017-03-23
    ========================================
    Date: 2017-03-24
    ========================================
    Date: 2017-03-25
    ========================================
    Date: 2017-03-26
    ========================================
    Date: 2017-03-27
    ========================================
    Date: 2017-03-28
    ========================================
    Date: 2017-03-29
    ========================================
    Date: 2017-03-30
    ========================================
    Date: 2017-03-31
    ========================================
    Date: 2017-04-01
    ========================================
    Date: 2017-04-02
    ========================================
    Date: 2017-04-03
    ========================================
    Date: 2017-04-04
    ========================================
    Date: 2017-04-05
    ========================================
    Date: 2017-04-06
    ========================================
    Date: 2017-04-07
    ========================================
    Date: 2017-04-08
    ========================================
    Date: 2017-04-09
    ========================================
    Date: 2017-04-10
    ========================================
    Date: 2017-04-11
    ========================================
    Date: 2017-04-12
    ========================================
    Date: 2017-04-13
    ========================================
    Date: 2017-04-14
    ========================================
    Date: 2017-04-15
    Mean ENMO for Sedentary: 18.186152384345696 mili-g
    Time Spent for Sedentary: 417.0833333333333 minutes
    Mean ENMO for Light Activity: 64.0617744069671 mili-g
    Time Spent for Light Activity: 144.66666666666666 minutes
    Mean ENMO for Moderate Activity: 133.31321107435284 mili-g
    Time Spent for Moderate Activity: 32.166666666666664 minutes
    Mean ENMO for Vigorous Activity: 545.4999246492408 mili-g
    Time Spent for Vigorous Activity: 0.08333333333333333 minutes
    ========================================
    Date: 2017-04-16
    Mean ENMO for Sedentary: 17.75617937313141 mili-g
    Time Spent for Sedentary: 273.8333333333333 minutes
    Mean ENMO for Light Activity: 63.20102933555767 mili-g
    Time Spent for Light Activity: 56.58333333333333 minutes
    Mean ENMO for Moderate Activity: 155.85012675969605 mili-g
    Time Spent for Moderate Activity: 15.916666666666666 minutes
    ========================================
    Date: 2017-04-17
    ========================================
    Date: 2017-04-18
    ========================================
    Date: 2017-04-19
    ========================================
    Date: 2017-04-20
    ========================================
    Date: 2017-04-21
    ========================================
    Date: 2017-04-22
    ========================================
    Date: 2017-04-23
    ========================================
    Date: 2017-04-24
    ========================================
    Date: 2017-04-25
    ========================================
    Date: 2017-04-26
    ========================================
    Date: 2017-04-27
    ========================================
    Date: 2017-04-28
    ========================================
    Date: 2017-04-29
    ========================================
    Date: 2017-04-30
    ========================================
    Date: 2017-05-01
    ========================================
    Date: 2017-05-02
    ========================================
    Date: 2017-05-03
    ========================================
    Date: 2017-05-04
    ========================================
    Date: 2017-05-05
    ========================================
    Date: 2017-05-06
    ========================================
    Date: 2017-05-07
    ========================================
    Date: 2017-05-08
    ========================================
    Date: 2017-05-09
    ========================================
    Date: 2017-05-10
    ========================================
    Date: 2017-05-11
    ========================================
    Date: 2017-05-12
    ========================================
    Date: 2017-05-13
    ========================================
    Date: 2017-05-14
    ========================================
    Date: 2017-05-15
    ========================================
    Date: 2017-05-16
    ========================================
    Date: 2017-05-17
    ========================================
    Date: 2017-05-18
    ========================================
    Date: 2017-05-19
    ========================================
    Date: 2017-05-20
    ========================================
    Date: 2017-05-21
    ========================================
    Date: 2017-05-22
    ========================================
    Date: 2017-05-23
    ========================================
    Date: 2017-05-24
    ========================================
    Date: 2017-05-25
    ========================================
    Date: 2017-05-26
    ========================================
    Date: 2017-05-27
    ========================================
    Date: 2017-05-28
    ========================================
    Date: 2017-05-29
    ========================================
    Date: 2017-05-30
    ========================================
    Date: 2017-05-31
    ========================================
    Date: 2017-06-01
    ========================================
    Date: 2017-06-02
    ========================================
    Date: 2017-06-03
    ========================================
    Date: 2017-06-04
    ========================================
    Date: 2017-06-05
    ========================================
    Date: 2017-06-06
    ========================================
    Date: 2017-06-07
    ========================================
    Date: 2017-06-08
    ========================================
    Date: 2017-06-09
    ========================================
    Date: 2017-06-10
    ========================================
    Date: 2017-06-11
    ========================================
    Date: 2017-06-12
    ========================================
    Date: 2017-06-13
    ========================================
    Date: 2017-06-14
    ========================================
    Date: 2017-06-15
    ========================================
    Date: 2017-06-16
    ========================================
    Date: 2017-06-17
    ========================================
    Date: 2017-06-18
    ========================================
    Date: 2017-06-19
    ========================================
    Date: 2017-06-20
    ========================================
    Date: 2017-06-21
    ========================================
    Date: 2017-06-22
    ========================================
    Date: 2017-06-23
    ========================================
    Date: 2017-06-24
    ========================================
    Date: 2017-06-25
    ========================================
    Date: 2017-06-26
    ========================================
    Date: 2017-06-27
    ========================================
    Date: 2017-06-28
    ========================================
    Date: 2017-06-29
    ========================================
    Date: 2017-06-30
    ========================================
    Date: 2017-07-01
    ========================================
    Date: 2017-07-02
    ========================================
    Date: 2017-07-03
    ========================================
    Date: 2017-07-04
    ========================================
    Date: 2017-07-05
    ========================================
    Date: 2017-07-06
    ========================================
    Date: 2017-07-07
    Mean ENMO for Sedentary: 26.128116315767418 mili-g
    Time Spent for Sedentary: 43.416666666666664 minutes
    Mean ENMO for Light Activity: 59.93098272874184 mili-g
    Time Spent for Light Activity: 13.0 minutes
    Mean ENMO for Moderate Activity: 126.59609977940086 mili-g
    Time Spent for Moderate Activity: 1.6666666666666665 minutes
    ========================================
    Date: 2017-07-08
    Mean ENMO for Sedentary: 20.402291249078385 mili-g
    Time Spent for Sedentary: 520.0 minutes
    Mean ENMO for Light Activity: 62.93458414555094 mili-g
    Time Spent for Light Activity: 166.16666666666666 minutes
    Mean ENMO for Moderate Activity: 135.92651383290314 mili-g
    Time Spent for Moderate Activity: 34.416666666666664 minutes
    ========================================
    Date: 2017-07-09
    ========================================
    Date: 2017-07-10
    ========================================
    Date: 2017-07-11
    ========================================
    Date: 2017-07-12
    ========================================
    Date: 2017-07-13
    ========================================
    Date: 2017-07-14
    ========================================
    Date: 2017-07-15
    ========================================
    Date: 2017-07-16
    ========================================
    Date: 2017-07-17
    ========================================
    Date: 2017-07-18
    Mean ENMO for Sedentary: 23.619923480209174 mili-g
    Time Spent for Sedentary: 551.0833333333333 minutes
    Mean ENMO for Light Activity: 64.5326807562732 mili-g
    Time Spent for Light Activity: 230.91666666666666 minutes
    Mean ENMO for Moderate Activity: 138.6372104484788 mili-g
    Time Spent for Moderate Activity: 64.75 minutes
    Mean ENMO for Vigorous Activity: 502.5867304927892 mili-g
    Time Spent for Vigorous Activity: 1.5 minutes
    ========================================
    Date: 2017-07-19
    ========================================
    Date: 2017-07-20
    Mean ENMO for Sedentary: 23.06080101968062 mili-g
    Time Spent for Sedentary: 630.3333333333333 minutes
    Mean ENMO for Light Activity: 62.47312156615242 mili-g
    Time Spent for Light Activity: 283.5833333333333 minutes
    Mean ENMO for Moderate Activity: 134.21407547133205 mili-g
    Time Spent for Moderate Activity: 47.0 minutes
    ========================================
    Date: 2017-07-21
    ========================================
    Date: 2017-07-22
    Mean ENMO for Sedentary: 21.2138077713461 mili-g
    Time Spent for Sedentary: 363.3333333333333 minutes
    Mean ENMO for Light Activity: 68.3158459768656 mili-g
    Time Spent for Light Activity: 193.5 minutes
    Mean ENMO for Moderate Activity: 157.3477545016603 mili-g
    Time Spent for Moderate Activity: 116.75 minutes
    Mean ENMO for Vigorous Activity: 453.4500635545964 mili-g
    Time Spent for Vigorous Activity: 1.5833333333333333 minutes
    ========================================
    Date: 2017-07-23
    ========================================
    Date: 2017-07-24
    ========================================
    Date: 2017-07-25
    Mean ENMO for Sedentary: 23.50782782374932 mili-g
    Time Spent for Sedentary: 535.9166666666666 minutes
    Mean ENMO for Light Activity: 65.53882490778041 mili-g
    Time Spent for Light Activity: 243.66666666666666 minutes
    Mean ENMO for Moderate Activity: 141.77375634504278 mili-g
    Time Spent for Moderate Activity: 99.41666666666666 minutes
    Mean ENMO for Vigorous Activity: 460.8174407081833 mili-g
    Time Spent for Vigorous Activity: 0.5 minutes
    ========================================
    Date: 2017-07-26
    Mean ENMO for Sedentary: 19.412527114618126 mili-g
    Time Spent for Sedentary: 597.5 minutes
    Mean ENMO for Light Activity: 63.34909749803663 mili-g
    Time Spent for Light Activity: 207.5 minutes
    Mean ENMO for Moderate Activity: 144.23413216301785 mili-g
    Time Spent for Moderate Activity: 70.16666666666666 minutes
    ========================================
    Date: 2017-07-27
    Mean ENMO for Sedentary: 22.44149976356279 mili-g
    Time Spent for Sedentary: 571.25 minutes
    Mean ENMO for Light Activity: 62.14773327227127 mili-g
    Time Spent for Light Activity: 264.91666666666663 minutes
    Mean ENMO for Moderate Activity: 138.30077036181876 mili-g
    Time Spent for Moderate Activity: 34.0 minutes
    ========================================
    
