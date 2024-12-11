# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Function to plot results of the gait sequence detection algorithm
def plot_gait(target_sampling_freq_Hz, detected_activity_signal, gait_sequences_):
    """
    Plot the detected gait sequences.

    Args:
        target_sampling_freq_Hz (float) : Target sampling frequency.
        detected_activity_signal (np.array): Pre-processed acceleration signal.
        gait_sequences_ (pd.DataFrame): Detected gait sequences.

    Returns:
        plot
    """
    plt.figure(figsize=(22, 14))

    plt.plot(
        np.arange(len(detected_activity_signal)) / (60 * target_sampling_freq_Hz),
        detected_activity_signal,
        label="Pre-processed acceleration signal",
    )
    plt.title("Detected gait sequences", fontsize=18)
    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Acceleration (m/s$^{2}$)", fontsize=14)

    # Fill the area between start and end times
    for index, sequence in gait_sequences_.iterrows():
        onset = sequence["onset"] / 60  # Convert to minutes
        end_time = (sequence["onset"] + sequence["duration"]) / 60  # Convert to minutes
        plt.axvline(onset, color="g")
        plt.axvspan(onset, end_time, facecolor="grey", alpha=0.8)
    plt.legend(
        ["Pre-processed acceleration signal", "Gait onset", "Gait duration"],
        fontsize=20,
        loc="best",
    )
    plt.grid(visible=None, which="both", axis="both")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


# Function to plot results of the physical activity monitoring algorithm
def plot_pam(hourly_average_data):
    """
    Plots the hourly averaged ENMO for each day along with the mean of all days.

    Args:
        hourly_average_data (pd.DataFrame): DataFrame containing hourly averaged ENMO values.
    """
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Choose the 'turbo' colormap for coloring each day
    colormap = plt.cm.turbo

    # Normalize the index for the colormap
    num_days = len(hourly_average_data.index)
    norm = plt.Normalize(vmin=0, vmax=num_days - 1)

    # Plot each day's data with a unique color
    for i, date in enumerate(hourly_average_data.index):
        color = colormap(norm(i))  # Normalize the index
        ax.plot(hourly_average_data.loc[date], label=str(date), color=color)

    # Calculate and plot the mean across all days
    mean_enmo = hourly_average_data.mean(axis=0)
    ax.plot(
        mean_enmo,
        label="Mean across all days",
        color="black",
        linestyle="--",
        linewidth=2,
    )

    # Customize plot
    ax.set_xticks(hourly_average_data.columns)
    ax.set_xticklabels(hourly_average_data.columns)
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel("ENMO (mg)", fontsize=14)
    plt.title("Hourly averaged ENMO for each day")
    plt.legend(loc="upper left", fontsize=14)
    plt.grid(visible=None, which="both", axis="both")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


# Function to plot results of the postural transition detection algorithm
def plot_postural_transitions(accel, gyro, postural_transitions_, sampling_freq_Hz):
    """
    Plot results of the gait sequence detection algorithm.

    Args:
        accel (ndarray): Array of acceleration data.
        gyro (ndarray): Array of gyroscope data.
        postural_transitions_ (DataFrame): DataFrame containing postural transition information.
        sampling_freq_Hz (float): Sampling frequency in Hertz.

    Returns:
        Plot postural transitions
    """
    # Convert acceleration data from "g" to "m/s^2"
    accel *= 9.81

    # Figure
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: ACCEL data
    ax1 = plt.subplot(211)
    for i in range(3):
        ax1.plot(
            np.arange(len(accel)) / sampling_freq_Hz,
            accel[:, i],
        )
    for i in range(len(postural_transitions_)):
        onset = postural_transitions_["onset"][i]
        duration = postural_transitions_["duration"][i]
        ax1.axvline(x=onset, color="r")
        ax1.axvspan(onset, (onset + duration), color="grey")
    ax1.set_ylabel(f"Acceleration (m/s$^{2}$)", fontsize=14)
    ax1.set_xlabel(f"Time (s)", fontsize=14)
    ax1.legend(
        ["ACCEL x", "ACCEL y", "ACCEL z", "Event oset", "Event duration"],
        loc="upper left",
        fontsize=14,
        framealpha=0.5,
    )
    accel_min = np.min(accel)
    accel_max = np.max(accel)
    buffer = (accel_max - accel_min) * 0.1
    ax1.set_ylim(accel_min - buffer, accel_max + buffer)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Subplot 2: GYRO data
    ax2 = plt.subplot(212)
    for i in range(3):
        ax2.plot(
            np.arange(len(gyro)) / sampling_freq_Hz,
            gyro[:, i],
        )
    for i in range(len(postural_transitions_)):
        onset = postural_transitions_["onset"][i]
        duration = postural_transitions_["duration"][i]
        ax2.axvline(x=onset, color="r")
        ax2.axvspan(onset, (onset + duration), color="grey")
    ax2.set_ylabel(f"Gyro (deg/s)", fontsize=14)
    ax2.set_xlabel(f"Time (s)", fontsize=14)
    ax2.legend(
        ["GYRO x", "GYRO y", "GYRO z", "Event oset", "Event duration"],
        loc="upper left",
        fontsize=14,
        framealpha=0.5,
    )
    gyro_min = np.min(gyro)
    gyro_max = np.max(gyro)
    buffer = (gyro_max - gyro_min) * 0.1
    ax2.set_ylim(gyro_min - buffer, gyro_max + buffer)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()


# Function to plot results of the turn detection algorithm
def plot_turns(accel, gyro, detected_turns, sampling_freq_Hz):
    """
    Plot results of the turn detection algorithm.

    Args:
        accel (ndarray): Array of acceleration data.
        gyro (ndarray): Array of gyroscope data.
        detected_turns (DataFrame): DataFrame containing detected turns information.
        sampling_freq_Hz (float): Sampling frequency in Hz.

    Returns:
        Plot detected turns on the data
    """
    # Convert acceleration data from "g" to "m/s^2"
    accel *= 9.81

    # Convert gyro data from "rad/s" to "deg/s"
    gyro = np.rad2deg(gyro)

    # Figure
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: ACCEL data
    ax1 = plt.subplot(211)
    for i in range(3):
        ax1.plot(
            np.arange(len(accel)) / sampling_freq_Hz,
            accel[:, i],
        )
    for i in range(len(detected_turns)):
        onset = detected_turns["onset"][i]
        duration = detected_turns["duration"][i]
        ax1.axvline(x=onset, color="r")
        ax1.axvspan(onset, (onset + duration), color="grey")
    ax1.set_ylabel(f"Acceleration (m/s$^{2}$)", fontsize=14)
    ax1.set_xlabel("Time (s)", fontsize=14)
    ax1.legend(
        ["ACCEL x", "ACCEL y", "ACCEL z", "Turn onset", "Turn duration"],
        loc="upper left",
        fontsize=14,
        framealpha=0.5,
    )
    accel_min = np.min(accel)
    accel_max = np.max(accel)
    buffer = (accel_max - accel_min) * 0.1
    ax1.set_ylim(accel_min - buffer, accel_max + buffer)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Subplot 2: GYRO data
    ax2 = plt.subplot(212)
    for i in range(3):
        ax2.plot(
            np.arange(len(gyro)) / sampling_freq_Hz,
            gyro[:, i],
        )
    for i in range(len(detected_turns)):
        onset = detected_turns["onset"][i]
        duration = detected_turns["duration"][i]
        ax2.axvline(x=onset, color="r")
        ax2.axvspan(onset, (onset + duration), color="grey")
    ax2.set_ylabel("Gyro (deg/s)", fontsize=14)
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.legend(
        ["GYRO x", "GYRO y", "GYRO z", "Turn onset", "Turn duration"],
        loc="upper left",
        fontsize=14,
        framealpha=0.5,
    )
    gyro_min = np.min(gyro)
    gyro_max = np.max(gyro)
    buffer = (gyro_max - gyro_min) * 0.1
    ax2.set_ylim(gyro_min - buffer, gyro_max + buffer)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    fig.tight_layout()
    plt.show()
