import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5
import pandas as pd
from pathlib import Path


def main() -> None:
    # Set the parameters
    subject_id = "pp006"
    task_name = "tug"
    tracksys = "imu"

    # Load the data
    raw_data_path = Path(f"/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata")
    file_name = f"sub-{subject_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"
    data_df = pd.read_csv(raw_data_path / f"sub-{subject_id}" / "motion" / file_name, sep="\t", header=0)
    channels_df = pd.read_csv(raw_data_path / f"sub-{subject_id}" / "motion" / file_name.replace("_motion", "_channels"), sep="\t", header=0)

    # Extract the sampling frequency
    sampling_freq_Hz = channels_df["sampling_frequency"].values[0].astype(float)
    units = {ch_type: channels_df[(channels_df["type"] == ch_type)]["units"].values[0] for ch_type in ["ACC", "ANGVEL"]}

    # Get the accelerometer and gyroscope data from the pelvis sensor
    tracked_point = "pelvis"
    data = data_df.loc[:, [f"{tracked_point}_{ch_type}_{xyz}" for ch_type in ["ACC", "ANGVEL"] for xyz in "xyz"]]

    # Convert to SI units, if necessary
    if units["ACC"] == "g":
        data.loc[:, [f"{tracked_point}_ACC_{xyz}" for xyz in "xyz"]] *= 9.81
    if units["ANGVEL"] == "deg/s":
        data.loc[:, [f"{tracked_point}_ANGVEL_{xyz}" for xyz in "xyz"]] *= np.pi / 180.0
    
    # Plot the data
    fig, axs = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
    axs[0].plot(np.arange(len(data)) / sampling_freq_Hz, data.loc[:, [f"{tracked_point}_ACC_{xyz}" for xyz in "xyz"]])
    axs[1].plot(np.arange(len(data)) / sampling_freq_Hz, data.loc[:, [f"{tracked_point}_ANGVEL_{xyz}" for xyz in "xyz"]])
    axs[0].set_ylabel("acceleration (m/s²)")
    axs[1].set_ylabel("angular velocity (rad/s)")
    axs[1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    main()