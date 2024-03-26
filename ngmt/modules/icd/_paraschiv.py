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
    This Paraschiv-Ionescu initial contact detection algorithm identifies initial contact in accelerometer data
    collected from a low back IMU sensor. The purpose of algorithm is to identify and characterize initial contacts
    within walking bouts.

    The algorithm takes accelerometer data as input, and the vertical acceleration component, and processes each
    specified gait sequence independently. The signal is first detrended and then low-pass filtered. The resulting
    signal is numerically integrated and differentiated using a Gaussian continuous wavelet transformation. The
    initial contact (IC) events are identified as the positive maximal peaks between successive zero-crossings.

    Finally, initial contacts information is provided as a DataFrame with columns `onset`, `event_type`,
    `tracking_systems`, and `tracked_points`.

    Attributes:
        target_sampling_freq_Hz (float): Target sampling frequency for resampling the data. Default is 40.
        event_type (str): Type of the detected event. Default is 'initial contact'.
        tracking_systems (str): Tracking systems used. Default is 'SU'.
        tracked_points (str): Tracked points on the body. Default is 'LowerBack'.

    Methods:
        detect(data, gait_sequences, sampling_freq_Hz):
            Detects initial contacts on the accelerometer signal.

    Examples:
        Find initial contacts based on the detected gait sequence

        >>> icd = ParaschivIonescuInitialContactDetection()
        >>> icd = icd.detect(data=acceleration_data, sampling_freq_Hz=100)
        >>> print(icd.initial_contacts_)
                onset   event_type       tracking_systems   tracked_points
            0   5       initial contact  SU                 LowerBack
            1   5.6     initial contact  SU                 LowerBack

    References:
        [1] Paraschiv-Ionescu et al. (2019). Locomotion and cadence detection using a single trunk-fixed accelerometer...

        [2] Paraschiv-Ionescu et al. (2020). Real-world speed estimation using single trunk IMU: methodological challenges...
    """

    def __init__(
        self,
        target_sampling_freq_Hz: float = 40.0,
        event_type: str = "initial contact",
        tracking_systems: str = "SU",
        tracked_points: str = "LowerBack",
    ):
        """
        Initializes the ParaschivIonescuInitialContactDetection instance.

        Args:
            target_sampling_freq_Hz (float, optional): Target sampling frequency for resampling the data. Default is 40.
            event_type (str, optional): Type of the detected event. Default is 'gait sequence'.
            tracking_systems (str, optional): Tracking systems used. Default is 'SU'.
            tracked_points (str, optional): Tracked points on the body. Default is 'LowerBack'.
        """
        self.target_sampling_freq_Hz = target_sampling_freq_Hz
        self.event_type = event_type
        self.tracking_systems = tracking_systems
        self.tracked_points = tracked_points
        self.initial_contacts_ = None

    def detect(
        self,
        data: pd.DataFrame,
        gait_sequences: pd.DataFrame,
        sampling_freq_Hz: float = 100,
    ) -> pd.DataFrame:
        """
        Detects initial contacts based on the input accelerometer data.

        Args:
            data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
            gait_sequences (pd.DataFrame): Gait sequence calculated using ParaschivIonescuGaitSequenceDetectionDataframe algorithm.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data.

            Returns:
                ParaschivIonescuInitialContactDetection: Returns an instance of the class.
                    The initial contacts information is stored in the 'initial_contacts_' attribute,
                    which is a pandas DataFrame in BIDS format with the following columns:
                        - onset: Initial contacts.
                        - event_type: Type of the event (default is 'gait sequence').
                        - tracking_systems: Tracking systems used (default is 'SU').
                        - tracked_points: Tracked points on the body (default is 'LowerBack').
        """
        # Check if data is empty
        if data.empty:
            self.initial_contacts_ = pd.DataFrame()
            return  # Return without performing further processing

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
            stop_index = int(
                sampling_freq_Hz * (gait_seq["onset"] + gait_seq["duration"]) - 1
            )
            accv_gait_seq = acc_vertical[start_index : stop_index + 2].to_numpy()

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs)
                initial_contacts_rel, _ = preprocessing.signal_decomposition_algorithm(
                    accv_gait_seq, sampling_freq_Hz
                )
                initial_contacts = gait_seq["onset"] + initial_contacts_rel

                gait_seq["IC"] = initial_contacts.tolist()

                # Append onsets to the all_onsets list
                all_onsets.extend(initial_contacts)

            except Exception as e:
                print(
                    "Signal decomposition algorithm did not run successfully. Returning an empty vector of initial contacts"
                )
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
        self.initial_contacts_ = pd.DataFrame(
            {
                "onset": all_onsets,
                "event_type": self.event_type,
                "tracking_systems": self.tracking_systems,
                "tracked_points": self.tracked_points,
            }
        )

        return self
