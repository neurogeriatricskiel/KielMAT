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

    # Font size for all text elements
    font_size = 14

    plt.plot(
        np.arange(len(detected_activity_signal)) / (60 * target_sampling_freq_Hz),
        detected_activity_signal,
        label="Pre-processed acceleration signal",
    )
    plt.title("Detected gait sequences", fontsize=18)
    plt.xlabel("Time (minutes)", fontsize=font_size)
    plt.ylabel("Acceleration (g)", fontsize=font_size)

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
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


# Function to plot results of the physical activity monitoring algorithm
def plot_pam(hourly_average_data, thresholds_mg):
    """
    Plots the hourly averaged ENMO for each day along with activity level thresholds.

    Args:
        hourly_average_data (pd.DataFrame): DataFrame containing hourly averaged ENMO values.
        thresholds_mg (dict): Dictionary containing threshold values for physical activity detection.
    """
    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Font size for all text elements
    font_size = 14

    # Choose the 'turbo' colormap for coloring each day
    colormap = plt.cm.turbo

    # Plot thresholds
    ax.axhline(
        y=thresholds_mg.get("sedentary_threshold", 45),
        color="y",
        linestyle="--",
        label="Sedentary threshold",
    )
    ax.axhline(
        y=thresholds_mg.get("light_threshold", 100),
        color="g",
        linestyle="--",
        label="Light physical activity threshold",
    )
    ax.axhline(
        y=thresholds_mg.get("moderate_threshold", 400),
        color="r",
        linestyle="--",
        label="Moderate physical activity threshold",
    )

    # Plot each day data with a different color
    for i, date in enumerate(hourly_average_data.index):
        color = colormap(i)
        ax.plot(hourly_average_data.loc[date], label=str(date), color=color)

    # Customize plot
    plt.xticks(range(24), [str(i).zfill(2) for i in range(24)])
    plt.xlabel("Time (h)", fontsize=font_size)
    plt.ylabel("ENMO (mg)", fontsize=font_size)
    plt.title("Hourly averaged ENMO for each day along with activity level thresholds")
    plt.legend(loc="upper left", fontsize=font_size)
    plt.grid(visible=None, which="both", axis="both")
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
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
    # Figure
    fig = plt.figure(figsize=(12, 6))

    # Font size for all text elements
    font_size = 14

    # Subplot 1: Acceleration data
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
    ax1.set_title("Detected Postural Transitions", fontsize=font_size)
    ax1.set_ylabel(f"Acceleration (g)", fontsize=font_size)
    ax1.set_xlabel(f"Time (sec)", fontsize=font_size)
    ax1.legend(
        ["Acc x", "Acc y", "Acc z", "Event oset", "Event duration"],
        loc="upper right",
        fontsize=font_size,
    )
    accel_min = np.min(accel)
    accel_max = np.max(accel)
    buffer = (accel_max - accel_min) * 0.1
    ax1.set_ylim(accel_min - buffer, accel_max + buffer)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Subplot 2: Gyro data
    ax2 = plt.subplot(212)
    for i in range(3):
        ax2.plot(
            np.arange(len(gyro)) / sampling_freq_Hz,
            -gyro[:, i],
        )
    for i in range(len(postural_transitions_)):
        onset = postural_transitions_["onset"][i]
        duration = postural_transitions_["duration"][i]
        ax2.axvline(x=onset, color="r")
        ax2.axvspan(onset, (onset + duration), color="grey")
    ax1.set_title("Detected Postural Transitions", fontsize=font_size)
    ax2.set_ylabel(f"Gyro (deg/s)", fontsize=font_size)
    ax2.set_xlabel(f"Time (sec)", fontsize=font_size)
    ax2.legend(
        ["Gyr x", "Gyr y", "Gyr z", "Event oset", "Event duration"],
        loc="upper right",
        fontsize=font_size,
    )
    gyro_min = np.min(gyro)
    gyro_max = np.max(gyro)
    buffer = (gyro_max - gyro_min) * 0.1
    ax2.set_ylim(gyro_min - buffer, gyro_max + buffer)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    fig.tight_layout()
    plt.show()


# Function to plot results of the turn detection algorithm
def plot_turns(accel, gyro, accel_unit, gyro_unit, detected_turns, sampling_freq_Hz):
    """
    Plot results of the turn detection algorithm.

    Args:
        accel (ndarray): Array of acceleration data.
        gyro (ndarray): Array of gyroscope data.
        accel_unit (str): Unit of acceleration data.
        gyro_unit (str): Unit of gyro data.
        detected_turns (DataFrame): DataFrame containing detected turns information.
        sampling_freq_Hz (float): Sampling frequency in Hz.

    Returns:
        Plot detected turns on the data
    """
    # Font size for all text elements
    font_size = 14

    # Figure
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Acceleration data
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
    ax1.set_ylabel(f"Acceleration ({accel_unit})", fontsize=font_size)
    ax1.set_xlabel("Time (s)", fontsize=font_size)
    ax1.legend(
        ["Acc x", "Acc y", "Acc z", "Turn onset", "Turn duration"],
        loc="upper right",
        fontsize=font_size,
    )
    accel_min = np.min(accel)
    accel_max = np.max(accel)
    buffer = (accel_max - accel_min) * 0.1
    ax1.set_ylim(accel_min - buffer, accel_max + buffer)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Subplot 2: Gyro data
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
    ax2.set_ylabel("Gyro (rad/s)", fontsize=font_size)
    ax2.set_xlabel("Time (s)", fontsize=font_size)
    ax2.legend(
        ["Gyr x", "Gyr y", "Gyr z", "Turn onset", "Turn duration"],
        loc="upper right",
        fontsize=font_size,
    )
    gyro_min = np.min(gyro)
    gyro_max = np.max(gyro)
    buffer = (gyro_max - gyro_min) * 0.1
    ax2.set_ylim(gyro_min - buffer, gyro_max + buffer)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    fig.tight_layout()
    plt.show()
