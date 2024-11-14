import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "lightgray"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5
import pandas as pd
import pywt
import scipy.signal
import scipy.integrate
import scipy.optimize
from pathlib import Path
from kielmat.preprocessing import ButterworthFilter, CwtFilter, IOE


MAP_CHANNEL_TYPES = {
    "ACC": "ACCEL",
    "ANGVEL": "GYRO",
}  # see: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html#restricted-keyword-list-for-channel-type


def sigmoid_fn(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * x + b / (1.0 + np.exp((c - x) / d))


def main() -> None:
    # Set the parameters
    subject_id = "pp006"
    task_name = "tug"
    tracksys = "imu"

    # Load the data
    raw_data_path = Path(
        f"/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata"
    )
    file_name = f"sub-{subject_id}_task-{task_name}_tracksys-{tracksys}_motion.tsv"
    data_df = pd.read_csv(
        raw_data_path / f"sub-{subject_id}" / "motion" / file_name, sep="\t", header=0
    )
    channels_df = pd.read_csv(
        raw_data_path
        / f"sub-{subject_id}"
        / "motion"
        / file_name.replace("_motion", "_channels"),
        sep="\t",
        header=0,
    )

    # Extract the sampling frequency
    sampling_freq_Hz = channels_df["sampling_frequency"].values[0].astype(float)
    units = {
        ch_type: channels_df[(channels_df["type"] == ch_type)]["units"].values[0]
        for ch_type in ["ACC", "ANGVEL"]
    }

    # Get the accelerometer and gyroscope data from the pelvis sensor
    tracked_point = "pelvis"
    data = data_df.loc[
        :,
        [
            f"{tracked_point}_{ch_type}_{xyz}"
            for ch_type in ["ACC", "ANGVEL"]
            for xyz in "xyz"
        ],
    ]

    # Convert to SI units, if necessary
    if units["ACC"] == "g":
        data.loc[:, [f"{tracked_point}_ACC_{xyz}" for xyz in "xyz"]] *= 9.81
    if units["ANGVEL"] == "deg/s":
        data.loc[:, [f"{tracked_point}_ANGVEL_{xyz}" for xyz in "xyz"]] *= np.pi / 180.0

    # # Put data in xarray format
    # da = DataArray(
    #     data=data.values,
    #     dims=["time", "channels"],
    #     coords={
    #         "time": np.arange(len(data)) / sampling_freq_Hz,
    #         "channels": [
    #             "_".join(col.split("_")[:-2]) + "_" + MAP_CHANNEL_TYPES[col.split("_")[-2]] + "_" + col.split("_")[-1]
    #             for col in data.columns
    #         ]
    #     })
    # da.attrs = {"sampling_freq_Hz": sampling_freq_Hz, "units": {"ACCEL": "m/s²", "GYRO": "rad/s"}}

    # -----------------------------------------------------------
    # Orientation estimation
    ioe = IOE()
    rotated_data = ioe.apply(data.values, sampling_freq_Hz=sampling_freq_Hz)
    # Plot the data
    iplot = False
    if iplot:
        fig, axs = plt.subplots(2, 2, figsize=(9, 4), sharex=True)
        axs[0, 0].plot(
            np.arange(len(data)) / sampling_freq_Hz,
            data.loc[:, [f"{tracked_point}_ACC_{xyz}" for xyz in "xyz"]],
        )
        axs[0, 1].plot(np.arange(len(data)) / sampling_freq_Hz, rotated_data[:, 0:3])
        axs[0, 0].axhline(9.81, color="k", linestyle="--", alpha=0.5)
        axs[0, 1].axhline(9.81, color="k", linestyle="--", alpha=0.5)
        axs[0, 1].sharey(axs[0, 0])
        axs[1, 0].plot(
            np.arange(len(data)) / sampling_freq_Hz,
            data.loc[:, [f"{tracked_point}_ANGVEL_{xyz}" for xyz in "xyz"]],
        )
        axs[1, 1].plot(np.arange(len(data)) / sampling_freq_Hz, rotated_data[:, 3:6])
        axs[1, 1].sharey(axs[1, 0])
        axs[0, 0].set_ylabel("acceleration (m/s²)")
        axs[1, 0].set_ylabel("angular velocity (rad/s)")
        axs[1, 0].set_xlabel("time (s)")
        axs[1, 1].set_xlabel("time (s)")
        plt.tight_layout()
        plt.show()
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # Extract vertical acceleration only, and low-pass filter
    acc_z = rotated_data[:, 2] - 9.81  # vertical acceleration

    # Apply a Butterworth filter
    lowpass_filter = ButterworthFilter(cutoff_freq_Hz=1.3, order=4, btype="lowpass")
    acc_z_filtered = lowpass_filter.apply(acc_z, sampling_freq_Hz=sampling_freq_Hz)

    # -----------------------------------------------------------
    # Identify candidate postural transitions
    # Apply a continuous wavelet transform filter
    wavelet_name = "gaus1"
    frequencies = np.geomspace(0.2, 2.0, num=65, endpoint=True)
    scales = pywt.frequency2scale(wavelet_name, frequencies) / (1.0 / sampling_freq_Hz)
    cwt_filter = CwtFilter(wavelet_name=wavelet_name)
    cwt_coefs, cwt_freqs = cwt_filter.apply(
        acc_z_filtered, sampling_freq_Hz=sampling_freq_Hz, scales=scales
    )
    A = np.sum(np.abs(cwt_coefs), axis=0)

    # Find peaks in the wavelet power
    idx_pks, _ = scipy.signal.find_peaks(
        A, height=0.25 * np.max(A), distance=int(2 * sampling_freq_Hz)
    )
    idx_local_max, _ = scipy.signal.find_peaks(acc_z_filtered, height=0.0)
    idx_local_min, _ = scipy.signal.find_peaks(-acc_z_filtered, height=0.0)

    # Take derivate of filtered acceleration signal
    acc_z_filtered_diff = np.diff(acc_z_filtered) / (1.0 / sampling_freq_Hz)
    candidate_pts = {"onset": [], "duration": [], "event_type": []}
    acc_threshold = 0.3
    for idx_pk in idx_pks:
        if acc_z_filtered_diff[idx_pk] > 0:
            event_type = "stand-to-sit"

            # Find the last local minimum before the power peak
            f = np.argwhere(idx_local_min < idx_pk)[:, 0]
            idx_last_local_min = idx_local_min[f[-1]]

            # Find where the filtered acceleration was above the negative threshold
            # before the last local minimum
            f = np.argwhere(acc_z_filtered[:idx_last_local_min] > -acc_threshold)[:, 0]
            idx_onset = f[-1]

            # Find the first local maximum after the power peak
            f = np.argwhere(idx_local_max > idx_pk)[:, 0]
            idx_first_local_max = idx_local_max[f[0]]

            # Find where the filtered acceleration was below the positive threshold
            # after the first local maximum
            f = np.argwhere(acc_z_filtered[idx_first_local_max:] < acc_threshold)[:, 0]
            idx_offset = idx_first_local_max + f[0]

        else:
            event_type = "sit-to-stand"

            # Find the last local maximum before the power peak
            f = np.argwhere(idx_local_max < idx_pk)[:, 0]
            idx_last_local_max = idx_local_max[f[-1]]

            # Find where the filtered acceleration was below the positive threshold
            # before the last local maximum
            f = np.argwhere(acc_z_filtered[:idx_last_local_max] < acc_threshold)[:, 0]
            idx_onset = f[-1]

            # Find the first local minimum after the power peak
            f = np.argwhere(idx_local_min > idx_pk)[:, 0]
            idx_first_local_min = idx_local_min[f[0]]

            # Find where the filtered acceleration was above the negative threshold
            # after the first local minimum
            f = np.argwhere(acc_z_filtered[idx_first_local_min:] > -acc_threshold)[:, 0]
            idx_offset = idx_first_local_min + f[0]

        # Store the candidate event
        candidate_pts["onset"].append(idx_onset)
        candidate_pts["duration"].append(idx_offset - idx_onset)
        candidate_pts["event_type"].append(event_type)

    # Put the candidate events in a DataFrame
    pts_df = pd.DataFrame(candidate_pts)

    # -----------------------------------------------------------
    # For each candidate event, calculate the vertical velocity by numerical integration

    # Plot the results
    fig, axs = plt.subplots(3, 1, figsize=(9, 4), sharex=True)
    for idx_row, row in pts_df.iterrows():
        for ax in axs:
            ax.axvspan(
                row["onset"] / sampling_freq_Hz,
                (row["onset"] + row["duration"]) / sampling_freq_Hz,
                color=(
                    "tab:purple" if row["event_type"] == "stand-to-sit" else "tab:pink"
                ),
                ec="none",
                alpha=0.2,
            )

        # Calculate the vertical velocity
        idx_onset = row["onset"]
        idx_offset = row["onset"] + row["duration"]
        vel_z = scipy.integrate.cumulative_trapezoid(
            acc_z_filtered[idx_onset:idx_offset], dx=1.0 / sampling_freq_Hz, initial=0.0
        )

        # Band-pass filter to reduce drift and integration errors
        bandpass_filter = ButterworthFilter(
            order=3, cutoff_freq_Hz=(0.1, 50.0), btype="bandpass"
        )
        vel_z_filtered = bandpass_filter.apply(vel_z, sampling_freq_Hz=sampling_freq_Hz)

        # Calculate the vertical position
        pos_z = scipy.integrate.cumulative_trapezoid(
            vel_z_filtered, dx=1.0 / sampling_freq_Hz, initial=0.0
        )

        # Fit a sigmoid function to the vertical position data
        popt, pcov = scipy.optimize.curve_fit(
            f=sigmoid_fn, xdata=np.arange(len(pos_z)), ydata=pos_z, p0=[0.0005, 0.3, 125, 12.0]
        )

        axs[1].plot(np.arange(idx_onset, idx_offset) / sampling_freq_Hz, pos_z, label="position")
        axs[1].plot(np.arange(idx_onset, idx_offset) / sampling_freq_Hz, sigmoid_fn(np.arange(len(pos_z)), *popt), label="fit")

    axs[0].plot(np.arange(len(acc_z)) / sampling_freq_Hz, acc_z, label="raw")
    axs[0].plot(
        np.arange(len(acc_z)) / sampling_freq_Hz, acc_z_filtered, label="filtered"
    )
    for lim in [-0.3, 0.3]:
        axs[0].axhline(lim, color="gray", linestyle="--", alpha=0.35)
    axs[0].set_ylabel("acc (m/s²)")
    axs[0].legend(frameon=False)
    axs[2].plot(np.arange(len(A)) / sampling_freq_Hz, A)
    axs[2].axhline(0.25 * np.max(A), color="gray", linestyle="--", alpha=0.35)
    axs[2].set_ylabel("power")
    axs[2].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()

    # # -----------------------------------------------------------
    # # # Fit a sigmoid function to the vertical position data
    # # xdata = np
    # # popt, pcv = scipy.optimize.curve_fit(
    # #     f=sigmoid_fn,
    # #     xdata
    # # )
    # # Calculate the vertical velocity
    # idx_onset = pts_df["onset"].iloc[1]
    # idx_offset = pts_df["onset"].iloc[1] + pts_df["duration"].iloc[1]
    # vel_z = scipy.integrate.cumulative_trapezoid(
    #     acc_z_filtered[idx_onset:idx_offset], dx=1.0 / sampling_freq_Hz, initial=0.0
    # )

    # # Band-pass filter to reduce drift and integration errors
    # bandpass_filter = ButterworthFilter(
    #     order=3, cutoff_freq_Hz=(0.1, 50.0), btype="bandpass"
    # )
    # vel_z_filtered = bandpass_filter.apply(vel_z, sampling_freq_Hz=sampling_freq_Hz)

    # # Calculate the vertical position
    # pos_z = scipy.integrate.cumulative_trapezoid(
    #     vel_z_filtered, dx=1.0 / sampling_freq_Hz, initial=0.0
    # )
    
    # xs = np.arange(idx_onset, idx_offset) - idx_onset
    # ys = pos_z

    

    # fig, ax = plt.subplots()
    # ax.plot(xs, ys, label="data")
    # ax.plot(xs, sigmoid_fn(xs, *popt), label="fit")
    # ax.set_xlabel("time (samples)")
    # ax.set_ylabel("position (m)")
    # ax.legend(frameon=False)
    # plt.show()
    return


if __name__ == "__main__":
    main()
