# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


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
def plot_pam(hourly_average_data, thresholds_mg):
    """
    Plots the hourly averaged ENMO for each day along with the mean of all days and activity level thresholds.

    Args:
        hourly_average_data (pd.DataFrame): DataFrame containing hourly averaged ENMO values.
        thresholds_mg (dict): Dictionary containing threshold values for physical activity detection.
    """
    # Create a colormap with as many colors as the number of days
    num_days = len(hourly_average_data.index)
    colors = plt.cm.turbo(np.linspace(0, 1, num_days))

    # Calculate the mean of all days
    mean_data = hourly_average_data.mean(axis=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each day's data with a unique color
    for i, (date, row) in enumerate(hourly_average_data.iterrows()):
        ax.plot(row.index, row.values, label=f"Day {date}", color=colors[i])

    # Plot the mean of all days
    ax.plot(mean_data.index, mean_data.values, label="Mean of All Days", color="black", linestyle="--", linewidth=2)

    # Customize plot
    ax.set_xticks(hourly_average_data.columns)
    ax.set_xticklabels(hourly_average_data.columns, rotation=45)
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel("ENMO (mg)", fontsize=14)
    plt.title("Hourly Averaged ENMO for Each Day", fontsize=16)
    plt.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))
    plt.grid(visible=True, which="both", axis="both", linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
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



# Function to plot results of the sleep analysis algorithm
def plot_sleep_analysis(
    vertical_accel: np.ndarray,
    nocturnal_rest: np.ndarray,
    posture: np.ndarray,
    theta: np.ndarray,
    sampling_frequency: int,
    dt_data: Optional[pd.Series] = None,
):
    """
    Plots sleep analysis results, including vertical acceleration, orientation angles (theta),
    nocturnal rest, and posture classifications.

    Args:
        vertical_accel (np.ndarray): Vertical acceleration values.
        nocturnal_rest (np.ndarray): Binary array indicating nocturnal rest periods.
        posture (np.ndarray): Full posture array for visualization.
        theta (np.ndarray): Orientation angle values.
        sampling_frequency (int): Sampling frequency in Hz.
        dt_data (pd.Series, optional): Time axis corresponding to samples.
    """
    # Determine the time axis
    if dt_data is not None:
        time = dt_data
    else:
        time = pd.Series(
            pd.date_range(
                start="00:00:00",
                periods=len(vertical_accel),
                freq=pd.DateOffset(seconds=1 / sampling_frequency)
            )
        )

    # Color map for different postures with names
    posture_colors = {
        0: ('gray', 'Upright'),    # Upright
        1: ('red', 'Back'),       # Back
        2: ('green', 'Right side'), # Right side
        3: ('blue', 'Left side'), # Left side
        4: ('yellow', 'Belly')    # Belly
    }

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Subplot 1: Vertical acceleration and posture with colored regions
    ax1.plot(time, vertical_accel, label="Vertical Acceleration", color="blue")
    ax1.plot(
        time,
        nocturnal_rest * np.max(vertical_accel),
        label="Detected Nocturnal Rest",
        color="black",
        linestyle="--",
    )
    
    # For each unique posture region, plot with the corresponding color in ax1
    for posture_val, (color, label) in posture_colors.items():
        ax1.fill_between(time, 0, vertical_accel.max(), where=(posture == posture_val), 
                        color=color, alpha=0.3, label=f"{label}")
        ax1.fill_between(time, 0, vertical_accel.min(), where=(posture == posture_val), 
                        color=color, alpha=0.3)

    ax1.set_ylabel("Vertical Acceleration (g)", fontsize=14)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.set_title("Sleep Analysis", fontsize=16)

    # Subplot 2: Theta (orientation angle) and postures
    ax2.plot(time, theta, label="Theta (Orientation Angle)", color="blue")
    ax2.plot(
        time,
        nocturnal_rest * np.max(theta),
        label="Detected Nocturnal Rest",
        color="black",
        linestyle="--",
    )
    
    # For each unique posture region, plot with the corresponding color in ax1
    for posture_val, (color, label) in posture_colors.items():
        ax2.fill_between(time, 0, theta.max(), where=(posture == posture_val), 
                        color=color, alpha=0.3, label=f"{label}")
        ax2.fill_between(time, 0, theta.min(), where=(posture == posture_val), 
                        color=color, alpha=0.3)

    ax2.set_ylabel("Orientaion Angle (deg)", fontsize=14)
    ax2.legend(loc="upper left", fontsize=12)

    # Formatting x-axis to show every 2 hours
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2)) 
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %H:%M"))

    plt.tight_layout()
    plt.show()