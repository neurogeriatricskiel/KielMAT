import numpy as np
import matplotlib.pyplot as plt
from ngmt.datasets import mobilised
from ngmt.modules.gsd import ParaschivIonescuGaitSequenceDetection
from ngmt.modules.icd import ParaschivIonescuInitialContactDetection

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
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6 * 1 / 2.54, 4 * 1 / 2.54))
        axs[0].plot(np.arange(acc_data.shape[0]) / fs, acc_data)
        axs[1].plot(np.arange(gyr_data.shape[0]) / fs, gyr_data)
        for ax in axs:
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel("")
        axs[0].set_ylim((-1, 2))
        axs[1].set_ylim((-200, 200))
        axs[1].set_xlim((1322.0, 1340.0))
        plt.tight_layout()
        plt.show()
    return


if __name__ == "__main__":
    main()
