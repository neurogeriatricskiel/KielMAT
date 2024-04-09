from ngmt.datasets import mobilised
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE_PATH = "/home/robbin/Datasets/Mobilise-D/rawdata/sub-3008/Free-living/data.mat"


def plot_gait_sequences(
    acc_data: pd.DataFrame, sampling_frequency_Hz: float, gait_sequences: pd.DataFrame
) -> None:
    fig, ax = plt.subplots()
    for _, (onset, duration) in gait_sequences[["onset", "duration"]].iterrows():
        ax.axvspan(onset, onset + duration, color="tab:pink", alpha=0.2, ec=None)
    ax.plot(np.arange(len(acc_data)) / sampling_frequency_Hz, acc_data, lw=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (g)")
    plt.tight_layout()
    plt.show()
    return


def main() -> None:
    data = mobilised.load_recording(
        file_name=FILE_PATH,
        tracking_systems=["SU", "SU_INDIP"],
        tracked_points={
            "SU": "LowerBack",
            "SU_INDIP": ["LowerBack", "LeftFoot", "RightFoot"],
        },
    )
    acc_data = data.data["SU"].loc[
        :, [c for c in data.data["SU"].columns if "_ACC" in c]
    ]
    fs = data.channels["SU"]["sampling_frequency"].iloc[0].astype(float)

    gsd = ParaschivIonescuGaitSequenceDetection(
        tracking_systems="SU", tracked_points="LowerBack"
    )
    gsd.detect(data=acc_data, sampling_freq_Hz=fs)
    plot_gait_sequences(
        acc_data=acc_data, sampling_frequency_Hz=fs, gait_sequences=gsd.gait_sequences_
    )
    return


if __name__ == "__main__":
    main()
