import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kielmat.datasets import keepcontrol
from kielmat.utils.kielmat_dataclass import KielMATRecording


def main() -> None:
    recording = keepcontrol.load_recording(
        id="pp002",
        task="walkPreferred",
        tracking_systems="omc",
    )

    # Get the marker position data
    marker_data = recording.data["omc"].loc[:, [c for c in recording.data["omc"].columns if not c.endswith("_err")]]

    # Only keep the markers that were attached to the subject
    skip_markers = [
        f"{se}_{i}_pos_{xyz}"
        for se in ["start", "end"]
        for i in range(1, 3)
        for xyz in ["x", "y", "z"]
    ]
    marker_data = marker_data.drop(skip_markers, axis=1)
    marker_labels = [c.split("_pos")[0] for c in marker_data.columns[::3]]

    mean_trajectory = pd.DataFrame({
        f"pos_{xyz}": np.nanmean(
            marker_data[[
                c for c in marker_data.columns if c.endswith(f"_pos_{xyz}")
            ]],
            axis=1
        )
        for xyz in ["x", "y", "z"]
    })

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for mrk in marker_labels:
        ax.plot(
            marker_data[f"{mrk}_pos_x"].iloc[0],
            marker_data[f"{mrk}_pos_y"].iloc[0],
            marker_data[f"{mrk}_pos_z"].iloc[0],
            ls="none", marker="o", ms=6,
            label=mrk,
        )
    ax.plot(mean_trajectory["pos_x"].iloc[0], mean_trajectory["pos_y"].iloc[0], mean_trajectory["pos_z"].iloc[0], ls="none", marker="s", ms=10, label="Body Center")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    plt.show()

    fig, axs = plt.subplots(3, 1, sharex=True)
    for i, xyz in enumerate(["x", "y", "z"]):
        for mrk in ["l_asis", "r_asis", "l_psis", "r_psis"]:
            axs[i].plot(marker_data[f"{mrk}_pos_{xyz}"], label=mrk)
        axs[i].plot(mean_trajectory[f"pos_{xyz}"], ls="--", label="Body Center")
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    main()