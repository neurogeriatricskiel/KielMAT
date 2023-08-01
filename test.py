import numpy as np
import matplotlib.pyplot as plt
import os
from ngmt.datasets import keepcontrol

def main():
    # User settings
    SUB_ID = "pp032"
    TASK_NAME = "walkPreferred"
    TRACKSYS = "imu"

    # Load data
    ds = keepcontrol.load_file(
        sub_id=SUB_ID,
        task_name=TASK_NAME,
        tracksys=TRACKSYS
    )

    # Get a list of unique tracked points
    tracked_points = [imu.tracked_point for imu in ds.devices]

    # Get IMU data from given tracked point
    tracked_point = "pelvis"
    imu = [imu for imu in ds.devices if tracked_point in imu.tracked_point][0]

    fig, axs = plt.subplots(len(imu.recordings), 1, sharex=True)
    for i in range(len(imu.recordings)):
        axs[i].plot(imu.recordings[i].data)
        axs[i].set_ylabel(f"{imu.recordings[i].type} ({imu.recordings[i].units})")
    plt.show()
    return

if __name__ == "__main__":
    main()