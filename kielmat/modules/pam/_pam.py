import pandas as pd
import numpy as np
from kielmat.utils import preprocessing
from kielmat.utils import viz_utils


class PhysicalActivityMonitoring:
    """
    The algortihm monitors physical activity levels based on accelerometer data. It determines the
    intensity level of physical activities based on accelerometer signals using the following steps:

    - Load Data: Includes a time index and accelerometer data (N, 3) for x, y, and z axes. The
        sampling frequency (sampling_freq_Hz) is in Hz, with a default value of 100. Thresholds
        (thresholds_mg) are provided as a dictionary containing threshold values for physical
        activity detection in mg unit. The epoch duration (epoch_duration_sec) is defined in
        seconds, with a default of 5 seconds. The last input is plot, which, if set to
        True, generates a plot showing the average Euclidean Norm Minus One (ENMO) per hour for
        each date. The default is True.

    - Preprocess Signal: Calculate the sample-level Euclidean norm (EN) of the acceleration
        signal. Apply a fourth-order Butterworth low-pass filter with a cut-off frequency of 20Hz
        to remove noise. Calculate the Euclidean Norm Minus One (ENMO) index and truncate negative
        values to zero. Convert the indices by multiplying them by 1000 to convert units from g to
        mg.

    - Classify Intensity: Classify the intensity of physical activities based on the calculated
        ENMO values using 5-second epochs. Thresholds for categorization are as follows: sedentary
        activity < 45 mg, light activity 45–100 mg, moderate activity 100–400 mg, vigorous activity
        > 400 mg.

    - Classify Activities: Classify different levels of activities and calculate the time spent
        on each activity level for each day. If `plot` is True, the function generates a
        plot showing the averaged ENMO values for each day.

    Attributes:
        physical_activities_ (pd.DataFrame): DataFrame containing physical activity information for each day.

    Methods:
        detect(data, sampling_freq_Hz, thresholds_mg, epoch_duration_sec, plot):
            Detects gait sequences on the accelerometer signal.

    Examples:
        >>> pam = PhysicalActivityMonitoring()
        >>> pam.detect(
                data=acceleration_data,
                acceleration_unit:"m/s^2",
                sampling_freq_Hz=100,
                thresholds_mg={
                    "sedentary_threshold": 45,
                    "light_threshold": 100,
                    "moderate_threshold": 400,
                },
                epoch_duration_sec=5,
                plot=True)
        >>> print(pam.physical_activities_)
                        sedentary_mean_mg  sedentary_time_min  light_mean_mg  light_time_min  moderate_mean_mg  moderate_time_min  vigorous_mean_mg  vigorous_time_min
        3/19/2018       23.48              733.08              60.78          72              146.2             21.58              730.34            0.58
        3/20/2018       27.16              753.83              57.06          102.25          137.26            7.92               737.9             0.42

    References:
        [1] Doherty, Aiden, et al. (2017). Large scale population assessment of physical activity using wrist-worn accelerometers...

        [2] Van Hees, Vincent T., et al. (2013). Separating movement and gravity components in an acceleration signal and implications...
    """

    def __init__(self):
        """
        Initializes the physical activity instance.
        """
        self.physical_activities_ = None

    def detect(
        self,
        data: pd.DataFrame,
        acceleration_unit: str,
        sampling_freq_Hz: float,
        thresholds_mg: dict[str, float] = {
            "sedentary_threshold": 45,
            "light_threshold": 100,
            "moderate_threshold": 400,
        },
        epoch_duration_sec: float = 5,
        plot: bool = True,
    ) -> pd.DataFrame:
        """
        Detects and classifies physical activity levels.

        Args:
            data (pd.DataFrame): Input data with time index and accelerometer data (N, 3) for x, y, and z axes.
            acceleration_unit (str): Unit of input acceleration data.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data (in Hertz).
            thresholds_mg (dict): Dictionary containing threshold values for physical activity detection.
            epoch_duration_sec (int): Duration of each epoch in seconds.
            plot (bool): If True, generates a plot showing the average Euclidean Norm Minus One (ENMO). Default is True.

        Returns:
            pd.DataFrame: Contains date, sedentary_mean_mg, sedentary_time_min, light_mean_mg, light_time_min,
                          moderate_mean_mg, moderate_time_min, vigorous_mean_mg, vigorous_time_min
        """
        # Error handling for invalid input data

        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame.")

        # Check if data has at least 3 columns
        if data.shape[1] < 3:
            raise ValueError("Input data must have at least 3 columns.")

        # Create a time index if data does not have a timestamp column
        if data.index.name != "timestamp" or not isinstance(
            data.index, pd.DatetimeIndex
        ):
            # Create a timestamp index with the correct frequency if not already present
            data.index = pd.date_range(
                start="2023-01-01 00:00:00",
                periods=len(data),
                freq=f"{1/sampling_freq_Hz}s",
            )
            data.index.name = "timestamp"

        # check if index column in named timestamp
        if data.index.name != "timestamp":
            raise ValueError("Index column must be named timestamp.")

        if not isinstance(sampling_freq_Hz, (int, float)) or sampling_freq_Hz <= 0:
            raise ValueError("Sampling frequency must be a positive float.")

        if not isinstance(thresholds_mg, dict):
            raise ValueError("Thresholds must be a dictionary.")

        if not isinstance(epoch_duration_sec, int) or epoch_duration_sec <= 0:
            raise ValueError("Epoch duration must be a positive integer.")

        if not isinstance(plot, bool):
            raise ValueError("Plot results must be a boolean (True or False).")

        # Check unit of acceleration data if it is in g or m/s^2
        if acceleration_unit == "m/s^2":
            # Convert acceleration data from m/s^2 to g (if not already is in g)
            data = data.copy()
            data /= 9.81

        # Calculate Euclidean Norm (EN)
        data = data.copy()
        data["en"] = np.linalg.norm(data.values, axis=1)

        # Apply 4th order low-pass Butterworth filter with the cutoff frequency of 20Hz
        data["en"] = preprocessing.lowpass_filter(
            data["en"].values,
            method="butter",
            order=4,
            cutoff_freq_hz=20,
            sampling_rate_hz=sampling_freq_Hz,
        )

        # Calculate Euclidean Norm Minus One (ENMO) value
        data["enmo"] = data["en"] - 1

        # Set negative values of ENMO to zero
        data["truncated_enmo"] = np.maximum(data["enmo"], 0)

        # Convert ENMO from g to milli-g
        data["enmo"] = data["truncated_enmo"] * 1000

        # Create a final DataFrame with time index and processed ENMO values
        processed_data = pd.DataFrame(data={"enmo": data["enmo"]}, index=data.index)

        # Classify activities based on thresholds using activity_classification
        classified_processed_data = preprocessing.classify_physical_activity(
            processed_data,
            time_column_name=data.index.name,
            sedentary_threshold=thresholds_mg.get("sedentary_threshold"),
            light_threshold=thresholds_mg.get("light_threshold"),
            moderate_threshold=thresholds_mg.get("moderate_threshold"),
            epoch_duration=epoch_duration_sec,
        )

        # Extract date from the datetime index
        classified_processed_data["date"] = classified_processed_data[
            data.index.name
        ].dt.date

        # Calculate time spent in each activity level for each epoch
        classified_processed_data["sedentary_time_min"] = (
            classified_processed_data["sedentary"] * epoch_duration_sec
        ) / 60
        classified_processed_data["light_time_min"] = (
            classified_processed_data["light"] * epoch_duration_sec
        ) / 60
        classified_processed_data["moderate_time_min"] = (
            classified_processed_data["moderate"] * epoch_duration_sec
        ) / 60
        classified_processed_data["vigorous_time_min"] = (
            classified_processed_data["vigorous"] * epoch_duration_sec
        ) / 60

        # Group by date and calculate mean and total time spent in each activity level
        physical_activities_ = (
            classified_processed_data.groupby("date")
            .agg(
                sedentary_mean_enmo=(
                    "enmo",
                    lambda x: (
                        np.nanmean(
                            x[classified_processed_data.loc[x.index, "sedentary"] == 1]
                        )
                        if len(
                            x[classified_processed_data.loc[x.index, "sedentary"] == 1]
                        )
                        > 0
                        else 0
                    ),
                ),
                sedentary_time_min=("sedentary_time_min", "sum"),
                light_mean_enmo=(
                    "enmo",
                    lambda x: (
                        np.nanmean(
                            x[classified_processed_data.loc[x.index, "light"] == 1]
                        )
                        if len(x[classified_processed_data.loc[x.index, "light"] == 1])
                        > 0
                        else 0
                    ),
                ),
                light_time_min=("light_time_min", "sum"),
                moderate_mean_enmo=(
                    "enmo",
                    lambda x: (
                        np.nanmean(
                            x[classified_processed_data.loc[x.index, "moderate"] == 1]
                        )
                        if len(
                            x[classified_processed_data.loc[x.index, "moderate"] == 1]
                        )
                        > 0
                        else 0
                    ),
                ),
                moderate_time_min=("moderate_time_min", "sum"),
                vigorous_mean_enmo=(
                    "enmo",
                    lambda x: (
                        np.nanmean(
                            x[classified_processed_data.loc[x.index, "vigorous"] == 1]
                        )
                        if len(
                            x[classified_processed_data.loc[x.index, "vigorous"] == 1]
                        )
                        > 0
                        else 0
                    ),
                ),
                vigorous_time_min=("vigorous_time_min", "sum"),
            )
            .reset_index()
        )

        # Return physical activities as an output
        self.physical_activities_ = physical_activities_

        # Group by date and hour to calculate the average ENMO for each hour
        hourly_average_data = processed_data.groupby(
            [processed_data.index.date, processed_data.index.hour]
        )["enmo"].mean()

        # Reshape the data to have dates as rows, hours as columns, and average ENMO as values
        hourly_average_data = hourly_average_data.unstack()

        self.hourly_average_data = hourly_average_data

        # Plot if set to true
        if plot:

            viz_utils.plot_pam(hourly_average_data)

        return self
