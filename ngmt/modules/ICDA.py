# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors


def Initial_Contact_Detection(
    imu_acceleration, gait_sequences, sampling_frequency=100, plot_results=False
):
    """
    Performs Initial Contact Detection (ICD) on low back IMU accelerometer data for detecting initial contacts (ICs).

    Args:
        imu_acceleration (numpy.ndarray): Input IMU accelerometer data (N, 3) for x, y, and z axes.
        gait_sequences (list): List of dictionaries containing gait sequence 'Start' and 'End' fields.
        sampling_frequency (float, optional): Sampling frequency of the accelerometer data. Default is 100 Hz.
        plot_results (bool, optional): If True, generates a plot showing the results. Default is False.

    Returns:
        processed_output (list): List of dictionaries containing detected ICs and associated information.

    Description:
        This function implements the Initial Contact Detection Algorithm (ICDA) on accelerometer data
        collected from a low back sensor. The purpose of ICD is to identify and characterize initial contacts
        within the walking bouts.

        The algorithm takes accelerometer data as input, specifically the vertical acceleration component,
        and processes each specified gait sequence independently. The input accelerometer data should be
        provided as a numpy.ndarray with shape (N, 3), where N is the number of data points.
        The three columns represent the acceleration along the x, y, and z axes. Gait sequences are provided
        as a list of dictionaries with 'Start' and 'End' fields indicating the start and end times of each sequence.
        The sampling frequency of the accelerometer data is also required.

        The algorithm is applied to a pre-processed vertical acceleration signal recorded on the lower back.
        This signal is first detrended and then low-pass filtered. The resulting signal is numerically integrated
        and differentiated using a Gaussian continuous wavelet transformation. The initial contact (IC) events
        are identified as the positive maximal peaks between successive zero-crossings.

    References:
        [1] McCamley, J., et al. (2012). An enhanced estimate of initial contact and final contact instants of
        time using lower trunk inertial sensor data. Gait & Posture, 36(2), 316-318.
        [2] Paraschiv-Ionescu, A, et al. "Real-world speed estimation using single trunk IMU: methodological
        challenges for impaired gait patterns." 2020 42nd Annual International Conference of the IEEE Engineering in
        Medicine & Biology Society (EMBC). IEEE, 2020.
        [3] Micó-Amigo, M. E., et al. (2022). Assessing real-world gait with digital technology? Validation,
        insights and recommendations from the Mobilise-D consortium.
        [4] Palmerini, L., et al. (2022) Mobility recorded by wearable devices and gold standards: the Mobilise-D
        procedure for data standardization. Scientific Data.
    """
    # Extract vertical accelerometer data
    acc_vertical = imu_acceleration[:, 0]

    # Process each gait sequence
    if isinstance(gait_sequences, list) and all(
        isinstance(seq, dict) and "Start" in seq and "End" in seq
        for seq in gait_sequences
    ):
        processed_output = []

        # Determine the maximum number of ICs among all gait sequences
        max_ic_count = max(len(gs.get("IC", [])) for gs in gait_sequences)

        for j in range(len(gait_sequences)):
            # Calculate start and stop indices for the current gait sequence
            start_index = int(sampling_frequency * gait_sequences[j]["Start"] - 1)
            stop_index = int(sampling_frequency * gait_sequences[j]["End"] - 1)
            accV_gs = acc_vertical[start_index : stop_index + 2]

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs)
                IC_rel, _ = preprocessing.signal_decomposition_algorithm(
                    accV_gs, sampling_frequency
                )
                Initial_Contact = (gait_sequences[j]["Start"]) + IC_rel

                gait_sequences[j]["IC"] = Initial_Contact.tolist()
            except Exception as e:
                print(
                    "SD algorithm did not run successfully. Returning an empty vector of ICs"
                )
                print(f"Error: {e}")
                Initial_Contact = []
                gait_sequences[j]["IC"] = []

            # Add sampling frequency and pad ICs to the output
            gait_sequences[j][
                "SamplingFrequency"
            ] = sampling_frequency  # Add sampling frequency to the output
            gait_sequences[j]["IC"] += [np.nan] * (
                max_ic_count - len(gait_sequences[j]["IC"])
            )

        processed_output = gait_sequences
    else:
        processed_output = []

    if plot_results:
        # Set the target sampling frequency for plotting
        target_sampling_frequency = 40
        max_ic_count = max(len(gs.get("IC", [])) for gs in gait_sequences)

        # Combine ICs from all gait sequences into a single array
        IC_all_signal = np.vstack(
            [
                gs.get("IC", []) + [np.nan] * (max_ic_count - len(gs.get("IC", [])))
                for gs in gait_sequences
                if "IC" in gs
            ]
        )

        # Convert ICs to sample indices
        IC_all_signal = np.nan_to_num(IC_all_signal)
        IC_indices = np.round(IC_all_signal * sampling_frequency).astype(int)

        # Resample and filter accelerometer data
        accV_resampled = preprocessing.resample_interpolate(
            acc_vertical, sampling_frequency, target_sampling_frequency
        )
        # Remove 40Hz drift from the filtered data
        drift_removed_accV = preprocessing.highpass_filter(
            signal=accV_resampled,
            sampling_frequency=target_sampling_frequency,
            method="iir",
        )

        # Load filter designed for low SNR, impaired, asymmetric and slow gait
        accV_filtered = preprocessing.lowpass_filter(drift_removed_accV, method="fir")
        accV_integral = (
            scipy.integrate.cumtrapz(accV_filtered) / target_sampling_frequency
        )

        # Perform Continuous Wavelet Transform (CWT)
        accV_cwt = preprocessing.apply_continuous_wavelet_transform(
            accV_integral,
            scales=9,
            desired_scale=9,
            wavelet="gaus2",
            sampling_frequency=target_sampling_frequency,
        )

        # Subtraction of the mean of the data from signal
        accV_cwt = accV_cwt - np.mean(accV_cwt)

        # Resample CWT results back to original sampling frequency
        accV_processed = preprocessing.resample_interpolate(
            accV_cwt, target_sampling_frequency, sampling_frequency
        )

        valid_IC_indices = IC_indices[
            (IC_indices >= 0) & (IC_indices < len(accV_processed))
        ]

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(
            np.arange(len(accV_processed)),
            accV_processed,
            linewidth=3,
            color=cfg_colors["raw"][0],
        )
        plt.plot(
            valid_IC_indices, accV_processed[valid_IC_indices], "ro", markersize=8
        )  # Plot ICs as red points
        plt.legend(
            ["Processed Vertical Acceleration Signal", "Initial Contacts (IC)"],
            fontsize=16,
        )
        plt.xlabel("Sample Index", fontsize=16)
        plt.ylabel("Amplitude", fontsize=16)
        plt.title(
            "Processed Acceleration Signal and Detected Initial Contacts", fontsize=16
        )
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

    return processed_output
