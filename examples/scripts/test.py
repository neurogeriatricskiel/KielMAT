from ngmt.datasets import mobilised
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


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


def plot_initial_contacts(
    acc_data: pd.DataFrame,
    sampling_frequency_Hz: float,
    v_acc_col_name: str,
    initial_contacts: pd.DataFrame,
    gait_sequences: Optional[pd.DataFrame] = None,
) -> None:
    idxs = initial_contacts["onset"].to_numpy() * sampling_frequency_Hz
    idxs = idxs.astype(int)
    fig, ax = plt.subplots()
    for _, (onset, duration) in gait_sequences[["onset", "duration"]].iterrows():
        ax.axvspan(onset, onset + duration, color="tab:pink", alpha=0.2, ec=None)
    ax.plot(np.arange(len(acc_data)) / sampling_frequency_Hz, acc_data, lw=2)
    ax.plot(
        initial_contacts["onset"].to_numpy(),
        acc_data[v_acc_col_name][idxs],
        ls="none",
        marker="^",
    )
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
    # plot_gait_sequences(
    #     acc_data=acc_data, sampling_frequency_Hz=fs, gait_sequences=gsd.gait_sequences_
    # )

    icd = ParaschivIonescuInitialContactDetection(
        tracking_systems="SU", tracked_points="LowerBack"
    )
    icd.detect(
        data=acc_data,
        sampling_freq_Hz=fs,
        v_acc_col_name=f"LowerBack_ACCEL_x",
        gait_sequences=gsd.gait_sequences_,
    )
    plot_initial_contacts(
        acc_data=acc_data,
        sampling_frequency_Hz=fs,
        v_acc_col_name="LowerBack_ACCEL_x",
        initial_contacts=icd.initial_contacts_,
        gait_sequences=gsd.gait_sequences_,
    )
    return


if __name__ == "__main__":
    main()
