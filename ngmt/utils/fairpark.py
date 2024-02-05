import os
import pandas as pd

def load_fairpark_data(data_folder_path):
    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate over all CSV files in the folder for each subject
    for file_name in os.listdir(data_folder_path):
        # Check if the file is a CSV file and has the expected prefix
        if file_name.endswith(".csv") and file_name.startswith("sub-023_imu-LARM_"):
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

    return combined_data, sampling_frequency
