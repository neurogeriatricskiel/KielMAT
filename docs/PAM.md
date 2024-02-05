# Physical Activity Monitoring (PAM)

This notebook will serve as a reference on how to use phyisical activity monitoring algortihm.

The example illustrates how the physical activity monitoring (PAM) algorithm is used to determine the intensity level of sedentary, light, moderate, and vigorous physical activities using body acceleration recorded with a triaxial accelerometer worn on the wrist. The physical activity monitoring algorithm is implemented in the main module `NGMT\ngmt\modules\PAM.py`.

The algorithm determines the intensity level of the physical activities based on the following steps. It starts by loading the input_data, which includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes. The other input of the algorithm is sampling_frequency, which is in Hz, and the default value is 100. Another input of the algorithm is thresholds, which should be provided as a dictionary containing threshold values for physical activity detection. Epoch length, which is defined in seconds, is the next input of the algorithm, as shown with epoch_duration. The default value is 5 seconds. The last input of the algorithm is plot_results, which, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each date. The default is True.

The following steps are taken to preprocess the input signal. First, the sample-level Euclidean norm (EN) of the acceleration signal across the x, y, and z axes is calculated. Next, a fourth-order Butterworth low-pass filter with a cut-off frequency of 20Hz is applied to remove noise. This filter is applied to the vector magnitude scores, rather than the individual axes. To separate out the activity-related component of the acceleration signal, the ENMO index is calculated. The Euclidean Norm Minus One (ENMO) is a summary metric for acceleration data and represents the vector magnitude of 3 axial measures minus the contribution of gravity (1 g). Then, negative values of the ENMO are truncated to zero. Finally, the calculated indices are multiplied by 1000 to convert the units of the acceleration from g to milli-g.

The algorithm continues by classifying the intensity of the physical activities based on the calculated ENMO values. Using the activity_classification function, the ENMO time-series data is then expressed in 5-second epochs. Epochs with a length of 5 seconds are used for summarizing the data, as this epoch length has been suggested to be able to capture shorter bouts of activities. The greater the intensity of movement and duration of activity in the summed 5-second epochs are, the greater the ENMO value is. Then, the intensity of activities as the time distribution of ENMO using 5-second epochs is used to classify activities based on different thresholds. In the analysis of intensity distribution, the following thresholds are used for categorization: sedentary activity < 45 milli-g, light activity 45–100 milli-g, moderate activity 100–400 milli-g, vigorous activity > 400 milli-g.

Finally, the algorithm takes the last steps to classify different levels of activities along with the time spent on each activity level for each day. The algorithm also visualizes the averaged ENMO values for each day.

::: modules.pam._pam
=======

## Import libraries
The necessary libraries such as pandas, os and physical activity monitoring (PHAM) are imported. Make sure that you have all the required libraries and modules installed before running this code. You also may need to install the 'ngmt' library and its dependencies if you haven't already.


```python
import pandas as pd
import numpy as np
from ngmt.modules import PHAM
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
Now, we are running physical activity monitoring (PHAM) algorithm from main module (`NGMT.ngmt.modules.PHAM.Physical_Activity_Monitoring`). The inputs of the algorihm are as follows. The input_data, includes data with a time index along with accelerometer data (N, 3) for x, y, and z axes. The sampling_frequency is the sampling frequecy of the acceleration data which is defined in Hz, and the default value of it is 100 Hz. The next input, thresholds, are provided as a dictionary containing threshold values for physical activity detection. The epoch_duration is the epoch length in seconds and the default value is 5 seconds. The last input of the algorithm is plot_results, which, if set to True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for each day. The default is True.





```python
# Use Physical_Activity_Monitoring to classify physical activity levels
phyam_results = PHAM.Physical_Activity_Monitoring(
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
    Date: 2017-07-20
    Mean ENMO for Sedentary: 23.06080101968062 mili-g
    Time Spent for Sedentary: 630.3333333333333 minutes
    Mean ENMO for Light Activity: 62.47312156615242 mili-g
    Time Spent for Light Activity: 283.5833333333333 minutes
    Mean ENMO for Moderate Activity: 134.21407547133205 mili-g
    Time Spent for Moderate Activity: 47.0 minutes
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
