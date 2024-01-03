import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ngmt.config import cfg_colors
from ngmt.utils import preprocessing


def Physical_Activity_Monitoring(
    input_data,
    sampling_frequency=100,
    thresholds={
        "sedentary_threshold": 45,
        "light_threshold": 100,
        "moderate_threshold": 400,
    },
    epoch_duration=5,
    plot_results=True,
):
    """
    Monitors physical activity levels based on accelerometer data.

    Args:
        input_data (numpy.ndarray): Input data with time index and accelerometer data (N, 3) for x, y, and z axes.
        sampling_frequency (float): Sampling frequency of the accelerometer data.
        thresholds (dict): Dictionary containing threshold values for physical activity detection.
        epoch_duration (int): Duration of each epoch in seconds.
        plot_results (bool, optional): If True, generates a plot showing the average Euclidean Norm Minus One (ENMO). Default is True.

    Description:
    The algorithm determines the intensity level of the phyisical activities based on following steps.
    It starts by loading the input_data which includes data with time index along with accelerometer data (N, 3) for x, y, and z axes.
    The other input of the algorithm is sampling_frequency which is in Hz and the default values is 100.
    Other input of the algorihtm is thresholds which should be provided as dictionary containing threshold values for physical activity detection.
    Epoch length which is defined in seconds is the next input of the algorihtm which is shown with epoch_duration.
    The default value is 5s. The last input of the algorithm is plot_results which if it set to True, the algorithm generates a plot showing the
    average Euclidean Norm Minus One (ENMO) per hour for each date. Default is True.

    The following steps are taken to preprocess the input signal. First, the sample level Euclidean norm (EN) of the acceleration signal
    across the x, y, and z axes is calculated. Next, a fourth order Butterworth low pass filter with a cut-off frequency of 20Hz is applied to
    remove noise. This filter is applied to the vector magnitude scores, rather than the individual axes. In order to separate out the
    activity-related component of the acceleration signal, the ENMO indice is calculted. The Euclidean Norm Minus One (ENMO) is a summary
    metric for acceleration data and represents the vector magnitude of 3 axial measures minus the contribution of gravity (1 g).
    Then, negative values of the ENMO are truncated to zero. Finally, the calculated indices are multiplied by 1000 to convert the units of
    the acceleration from g to mili-g.

    The algorithm continues by classifying the intensity of the physical activties based on the calculated ENMO values.
    Uisng the activity_classification function, the ENMO time-series data were then expressed in 5s epochs. Epochs with the length of 5s is used
    for summarizing the data as this epoch length has been suggested to be able to capture shorter bouts of activities.
    The greater the intensity of movement and duration of activity in the summed 5-second epochs are, the greater the ENMO value is.
    Then, the intensity of activities as the time distribution of ENMO using 5s epochs is used to classify activities based on different threshold.
    In the analysis of intensity distribution, the following thresholds are used for categorization:

    sedentary activity < 45 mili-g,
    light activity 45–100 mili-g,
    moderate activity 100–400 mili-g,
    vigorous activity > 400 mili-g.

    Finally, the algorithm takes the last steps to classify different levels of activties along with the time spent on each activity levels for each day.
    The algorithm also visulises the averaged ENMO values for each day.

    Returns:
        DataFrame (pandas.core.frame): Contains date, sedentary_mean_acc (mili-g), sedentary_spent_time_minute (min),
                                         light_mean_acc (mili-g), light_spent_time_minute (min),
                                         moderate_mean_acc (mili-g), moderate_spent_time_minute (min),
                                         vigorous_mean_acc (mili-g), vigorous_spent_time_minute (min)
    """
    # Error handling for invalid input data
    if not isinstance(sampling_frequency, (int, float)) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be a positive float.")

    if not isinstance(thresholds, dict):
        raise ValueError("Thresholds must be a dictionary.")

    if not isinstance(epoch_duration, int) or epoch_duration <= 0:
        raise ValueError("Epoch duration must be a positive integer.")

    if not isinstance(plot_results, bool):
        raise ValueError("Plot results must be a boolean (True or False).")

    # Calculate Euclidean Norm (EN)
    input_data["en"] = np.linalg.norm(input_data[["Acc_x", "Acc_y", "Acc_z"]], axis=1)

    # Apply 4th order low-pass Butterworth filter with the cutoff frequency of 20Hz
    input_data["en"] = preprocessing.lowpass_filter(
        input_data["en"].values,
        method="butter",
        order=4,
        cutoff_freq_hz=20,
        sampling_rate_hz=sampling_frequency,
    )

    # Calculate Euclidean Norm Minus One (ENMO) value
    input_data["enmo"] = input_data["en"] - 1

    # Set negative values of ENMO to zero
    input_data["truncated_enmo"] = np.maximum(input_data["enmo"], 0)

    # Convert ENMO from g to mili-g
    input_data["acc"] = input_data["truncated_enmo"] * 1000

    # Create a final DataFrame with time index and processed ENMO values
    processed_data = pd.DataFrame(
        data=input_data["acc"], index=input_data.index, columns=["acc"]
    )

    # Classify activities based on thresholds using activity_classification
    classified_processed_data = preprocessing.classify_physical_activity(
        processed_data,
        sedentary_threshold=thresholds.get("sedentary_threshold", 45),
        light_threshold=thresholds.get("light_threshold", 100),
        moderate_threshold=thresholds.get("moderate_threshold", 400),
        epoch_duration=epoch_duration,
    )

    # Plot results if set to true
    if plot_results:
        # Group by date and hour to calculate the average ENMO for each hour
        hourly_average_data = processed_data.groupby(
            [processed_data.index.date, processed_data.index.hour]
        )["acc"].mean()

        # Reshape the data to have dates as rows, hours as columns, and average ENMO as values
        hourly_average_data = hourly_average_data.unstack()

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 8))

        # Choose the 'turbo' colormap for coloring each day
        colormap = plt.cm.get_cmap("turbo", len(hourly_average_data.index))

        # Plot thresholds
        ax.axhline(
            y=thresholds.get("sedentary_threshold", 45),
            color="y",
            linestyle="--",
            label="Sedentary threshold",
        )
        ax.axhline(
            y=thresholds.get("light_threshold", 100),
            color="g",
            linestyle="--",
            label="Light physical activity threshold",
        )
        ax.axhline(
            y=thresholds.get("moderate_threshold", 400),
            color="r",
            linestyle="--",
            label="Moderate physical activity threshold",
        )

        # Plot each day data with a different color
        for i, date in enumerate(hourly_average_data.index):
            color = colormap(i)
            ax.plot(hourly_average_data.loc[date], label=str(date), color=color)

        # Customize plot appearance
        plt.xticks(range(24), [str(i).zfill(2) + ":00" for i in range(24)], rotation=45)
        plt.xlabel("Time (h)")
        plt.ylabel("Acceleration (mili-g)")
        plt.title("Hourly Averaged Euclidean Norm Minus One (ENMO) For Each Day")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

    # Extract date from the datetime index
    classified_processed_data["date"] = classified_processed_data["time"].dt.date

    # Calculate time spent in each activity level for each epoch
    classified_processed_data["sedentary_spent_time_minute"] = (
        classified_processed_data["sedentary"] * epoch_duration
    ) / 60
    classified_processed_data["light_spent_time_minute"] = (
        classified_processed_data["light"] * epoch_duration
    ) / 60
    classified_processed_data["moderate_spent_time_minute"] = (
        classified_processed_data["moderate"] * epoch_duration
    ) / 60
    classified_processed_data["vigorous_spent_time_minute"] = (
        classified_processed_data["vigorous"] * epoch_duration
    ) / 60

    # Group by date and calculate mean and total time spent in each activity level
    daily_classified_processed_data = (
        classified_processed_data.groupby("date")
        .agg(
            sedentary_mean_acc=(
                "acc",
                lambda x: np.mean(x[classified_processed_data["sedentary"] == 1]),
            ),
            sedentary_spent_time_minute=("sedentary_spent_time_minute", "sum"),
            light_mean_acc=(
                "acc",
                lambda x: np.mean(x[classified_processed_data["light"] == 1]),
            ),
            light_spent_time_minute=("light_spent_time_minute", "sum"),
            moderate_mean_acc=(
                "acc",
                lambda x: np.mean(x[classified_processed_data["moderate"] == 1]),
            ),
            moderate_spent_time_minute=("moderate_spent_time_minute", "sum"),
            vigorous_mean_acc=(
                "acc",
                lambda x: np.mean(x[classified_processed_data["vigorous"] == 1]),
            ),
            vigorous_spent_time_minute=("vigorous_spent_time_minute", "sum"),
        )
        .reset_index()
    )

    return daily_classified_processed_data
