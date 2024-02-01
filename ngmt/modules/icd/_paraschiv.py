# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from ngmt.utils import preprocessing
from ngmt.config import cfg_colors

class ParaschivIonescuInitialContactDetection:
    """
    Performs Initial Contact Detection (ICD) on low back IMU accelerometer data for detecting initial contacts (ICs).

    Args:
        data (numpy.ndarray): Input IMU accelerometer data (N, 3) for x, y, and z axes.
        gait_sequences (pd.DataFrame): DataFrame containing gait sequence information in BIDS format.
        sampling_frequency (float, optional): Sampling frequency of the accelerometer data. Default is 100 Hz.
        plot_results (bool, optional): If True, generates a plot showing the results. Default is False.

    Returns:
        processed_output (pd.DataFrame): DataFrame containing detected ICs and associated information.

    Description:
        This function implements the Initial Contact Detection Algorithm (ICDA) on accelerometer data
        collected from a low back sensor. The purpose of ICD is to identify and characterize initial contacts
        within the walking bouts.

        The algorithm takes accelerometer data as input, specifically the vertical acceleration component,
        and processes each specified gait sequence independently. The input accelerometer data should be
        provided as a numpy.ndarray with shape (N, 3), where N is the number of data points.
        The three columns represent the acceleration along the x, y, and z axes. Gait sequences are provided
        as a DataFrame with columns 'onset', 'duration', 'event_type', 'tracking_systems', and 'tracked_points'.
        The sampling frequency of the accelerometer data is also required.

        The algorithm is applied to a pre-processed vertical acceleration signal recorded on the lower back.
        This signal is first detrended and then low-pass filtered. The resulting signal is numerically integrated
        and differentiated using a Gaussian continuous wavelet transformation. The initial contact (IC) events
        are identified as the positive maximal peaks between successive zero-crossings.
    
    Methods:
        detect(data, sampling_freq_Hz, plot_results=False):
            Detects initial contacts based on the input accelerometer data.

            Args:
                data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
                gait_sequences (pd.DataFrame): Gait sequence calculated using ParaschivIonescuGaitSequenceDetectionDataframe algorithm.
                sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
                plot_results (bool, optional): If True, generates a plot showing the pre-processed acceleration data
                    and the detected gait sequences. Default is False.

            Returns:
                self (pd.DataFrame): DataFrame containing initial contact information in BIDS format.
    """
    def __init__(
        self,
        target_sampling_freq_Hz: float = 40.,
        event_type: str = "initial contact",
        tracking_systems: str = "SU",
        tracked_points: str = "LowerBack",
    ):
        self.target_sampling_freq_Hz = target_sampling_freq_Hz
        self.event_type = event_type
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.initial_contacts_ = None 

    def detect(
        self,
        data: pd.DataFrame,
        gait_sequences : pd.DataFrame,
        sampling_freq_Hz: float = 100,
        plot_results: bool = False
    ) -> pd.DataFrame:
        """
        Detects initial contacts based on the input accelerometer data.

        Args:
            data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
            gait_sequences (pd.DataFrame): Gait sequence calculated using ParaschivIonescuGaitSequenceDetectionDataframe algorithm.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
            plot_results (bool, optional): If True, generates a plot showing the pre-processed acceleration data
                and the detected initial contacts. Default is False.

        Returns:
            self (pd.DataFrame): DataFrame containing initial contact information in BIDS format.
        """
        # Extract vertical accelerometer data
        acc_vertical = data["LowerBack_ACCEL_x"]

        # Initialize an empty list to store the processed output
        processed_output = []

        # Initialize an empty list to store all onsets
        all_onsets = []

        # Process each gait sequence
        for _, gait_seq in gait_sequences.iterrows():
            # Calculate start and stop indices for the current gait sequence
            start_index = int(sampling_freq_Hz * gait_seq["onset"] - 1)
            stop_index = int(sampling_freq_Hz * (gait_seq["onset"] + gait_seq["duration"]) - 1)
            accv_gait_seq = acc_vertical[start_index: stop_index + 2].to_numpy()

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs)
                initial_contacts_rel, _ = preprocessing.signal_decomposition_algorithm(
                    accv_gait_seq, sampling_freq_Hz
                )
                initial_contacts = gait_seq["onset"] + initial_contacts_rel

                gait_seq["IC"] = initial_contacts.tolist()
                #print("IC for gait sequence:", Initial_Contact)
                #print("Length of IC for gait sequence:", len(gs["IC"]))

                # Append onsets to the all_onsets list
                all_onsets.extend(initial_contacts)

            except Exception as e:
                print("Signal decomposition algorithm did not run successfully. Returning an empty vector of initial contacts")
                print(f"Error: {e}")
                initial_contacts = []
                gait_seq["IC"] = []

            # Append the information to the processed_output list
            processed_output.append(gait_seq)

        # Check if processed_output is not empty
        if not processed_output:
            print("No initial contacts detected.")
            return pd.DataFrame()

        # Create a DataFrame from the processed_output list
        initial_contacts_ = pd.DataFrame(processed_output)

        # Create a BIDS-compatible DataFrame with all onsets
        self.initial_contacts_ = pd.DataFrame({"onset": all_onsets, 
                                    "event_type": self.event_type,
                                    "tracking_systems": self.tracking_systems,
                                    "tracked_points": self.tracked_points})

        if plot_results:
            # Combine ICs from all gait sequences into a single array
            max_ic_count = max(len(ic_list) for ic_list in initial_contacts_["IC"])
            initial_contacts_all_signals = np.full((len(initial_contacts_), max_ic_count), np.nan)

            # Fill IC_all_signal with ICs from each gait sequence
            for i, ic_list in enumerate(initial_contacts_["IC"]):
                initial_contacts_all_signals[i, :len(ic_list)] = ic_list

            # Convert ICs to sample indices
            initial_contacts_all_signals = np.nan_to_num(initial_contacts_all_signals)
            ic_indices = np.round(initial_contacts_all_signals * sampling_freq_Hz).astype(int)

            # Convert ICs to sample indices
            initial_contacts_all_signals = np.nan_to_num(initial_contacts_all_signals)
            ic_indices = np.round(initial_contacts_all_signals * sampling_freq_Hz).astype(int)

            # Resample and filter accelerometer data
            accv_resampled = preprocessing.resample_interpolate(
                acc_vertical.to_numpy(), sampling_freq_Hz, self.target_sampling_freq_Hz
            )

            # Remove 40Hz drift from the filtered data
            accv_drift_removed = preprocessing.highpass_filter(
                signal=accv_resampled,
                sampling_frequency=self.target_sampling_freq_Hz,
                method="iir",
            )

            # Load filter designed for low SNR, impaired, asymmetric and slow gait
            accv_filtered = preprocessing.lowpass_filter(
                accv_drift_removed, method="fir"
            )
            accv_integral = (
                scipy.integrate.cumtrapz(accv_filtered)
                / self.target_sampling_freq_Hz
            )

            # Perform Continuous Wavelet Transform (CWT)
            accv_cwt = preprocessing.apply_continuous_wavelet_transform(
                accv_integral,
                scales=9,
                desired_scale=9,
                wavelet="gaus2",
                sampling_frequency=self.target_sampling_freq_Hz,
            )

            # Subtraction of the mean of the data from signal
            accv_cwt = accv_cwt - np.mean(accv_cwt)

            # Resample CWT results back to original sampling frequency
            accv_processed = preprocessing.resample_interpolate(
                accv_cwt, self.target_sampling_freq_Hz, sampling_freq_Hz
            )

            valid_ic_indices = ic_indices[
                (ic_indices >= 0) & (ic_indices < len(accv_processed))
            ]

            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot(
                np.arange(len(accv_processed)),
                accv_processed,
                linewidth=3,
                color=cfg_colors["raw"][0],
            )
            plt.plot(
                valid_ic_indices, accv_processed[valid_ic_indices], "ro", markersize=8
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

        return self