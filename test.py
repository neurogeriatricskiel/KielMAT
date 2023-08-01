import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List
from dataclasses import dataclass, field
from ngmt.datasets import keepcontrol

@dataclass(kw_only=True)
class IMURecording:
    type: str
    units: str
    fs: float
    data: np.array

@dataclass(kw_only=True)
class IMUDevice:
    tracked_point: str
    device_manufacturer: Optional[str] = ""
    device_model: Optional[str] = ""
    recordings: List[IMURecording] = field(default_factory=list)

def main():
    # User settings
    SUB_ID = "pp032"
    TASK_NAME = "walkPreferred"
    TRACKSYS = "imu"

    # Load data
    df = keepcontrol.load_file(
        os.path.join(f"sub-{SUB_ID}", 
                     "motion",
                     f"sub-{SUB_ID}_task-{TASK_NAME}_tracksys-{TRACKSYS}_motion.tsv")
    )

    tracked_point = "sternum"
    acc = IMURecording(type="acc", units="g", fs=200., data=df.loc[:, [channel_name for channel_name in df.columns if f"{tracked_point}_ACC" in channel_name]].values)
    gyr = IMURecording(type="gyr", units="deg/s", fs=200., data=df.loc[:, [channel_name for channel_name in df.columns if f"{tracked_point}_ANGVEL" in channel_name]].values)
    mag = IMURecording(type="mag", units="a.u.", fs=200., data=df.loc[:, [channel_name for channel_name in df.columns if f"{tracked_point}_MAGN" in channel_name]].values)
    imu = IMUDevice(tracked_point=tracked_point, recordings=[acc, gyr, mag])

    fig, axs = plt.subplots(3, 1, sharex=True)
    for i in range(len(imu.recordings)):
        axs[i].plot(imu.recordings[i].data)
    axs[-1].set_xlim((0, acc.data.shape[0]))
    for ax in axs:
        ax.grid(True)
    plt.show()
    return

if __name__ == "__main__":
    main()