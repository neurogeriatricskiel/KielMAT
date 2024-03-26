import numpy as np
import matplotlib.pyplot as plt
from ngmt.datasets import mobilised
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection

plt.rcParams.update(
    {f"axes.spines.{which}": False for which in ["top", "right", "bottom", "left"]}
)

# from ngmt.config import cfg_colors

FILE_PATH = "/mnt/neurogeriatrics_data/Mobilise-D/rawdata/sub-4005/Free-living/data.mat"
TRACKSYS = "SU"
TRACKED_POINTS = {TRACKSYS: ["LowerBack"]}


def main() -> None:
    # Load recording
    rec = mobilised.load_recording(
        FILE_PATH, tracking_systems=TRACKSYS, tracked_points=TRACKED_POINTS
    )

    # Extract sensor data and sample rate
    acc_data = rec.data[TRACKSYS][
        [f"{TRACKED_POINTS[TRACKSYS][0]}_ACCEL_{x}" for x in ["x", "y", "z"]]
    ]
    gyr_data = rec.data[TRACKSYS][
        [f"{TRACKED_POINTS[TRACKSYS][0]}_GYRO_{x}" for x in ["x", "y", "z"]]
    ]
    fs = (
        rec.channels[TRACKSYS][
            rec.channels[TRACKSYS]["name"] == f"{TRACKED_POINTS[TRACKSYS][0]}_ACCEL_x"
        ]["sampling_frequency"]
        .values[0]
        .astype(float)
    )

    # Detect gait sequences
    gsd = ParaschivIonescuGaitSequenceDetection(
        tracking_systems=TRACKSYS, tracked_points=TRACKED_POINTS[TRACKSYS][0]
    )
    gsd = gsd.detect(acc_data, sampling_freq_Hz=fs)

    # Detect initial contacts
    icd = ParaschivIonescuInitialContactDetection(
        tracking_systems=TRACKSYS, tracked_points=TRACKED_POINTS[TRACKSYS][0]
    )
    icd = icd.detect(acc_data, gait_sequences=gsd.gait_sequences_, sampling_freq_Hz=fs)

    # Plot
    iplot = True
    if iplot:
        fig, ax = plt.subplots(figsize=(6 * 1 / 2.54, 3.5 * 1 / 2.54))
        ax.plot(np.arange(acc_data.shape[0]) / fs, acc_data)
        for _, (onset, duration) in gsd.gait_sequences_[
            ["onset", "duration"]
        ].iterrows():
            ax.axvspan(onset, onset + duration, color="tab:pink", ec="none", alpha=0.1)
        ax.set_xlim((1318.0, 1342.0))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig("fig_gs_example.png", dpi=300)
    return


if __name__ == "__main__":
    main()
