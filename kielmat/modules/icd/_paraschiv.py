# Import libraries
from typing import Optional
import pandas as pd
import scipy.signal
from kielmat.utils import preprocessing


class ParaschivIonescuInitialContactDetection:
    """
    This Paraschiv-Ionescu initial contact detection algorithm identifies initial contact in accelerometer data
    collected from a low back IMU sensor. The purpose of algorithm is to identify and characterize initial contacts
    within walking bouts.

    The algorithm takes accelerometer data as input, and the vertical acceleration component, and processes each
    specified gait sequence independently. The signal is first detrended and then low-pass filtered. The resulting
    signal is numerically integrated and differentiated using a Gaussian continuous wavelet transformation. The
    initial contact (IC) events are identified as the positive maximal peaks between successive zero-crossings. In 
    addition, final contact (FC) are identified are obtained from a further CWT differentiation of the signal [3].

    Finally, initial contacts information is provided as a DataFrame with columns `onset`, `event_type`, and
    `tracking_systems`. In addition, the Final contact inofrmation also provided as a DataFrame with columns `onset`, 
    `event_type`, and `tracking_systems`. 

    Methods:
        detect(accel_data, gait_sequences, sampling_freq_Hz):
            Detects initial contacts on the accelerometer signal.

    Examples:
        Detect initial contacts, final contacts, and calculate different gait parameters:

        >>> icd = ParaschivIonescuInitialContactDetection()
        >>> icd = icd.detect(accel_data=acceleration_data, sampling_freq_Hz=100)
        >>> print(icd.initial_contacts_)
                onset   event_type       duration   tracking_systems
            0   5       initial contact  0          SU
            1   5.6     initial contact  0          SU

        >>> print(icd.final_contacts_)
               onset       event_type       duration   tracking_systems
            0  5.300  final contact           0         SU
            1  6.100  final contact           0         SU

    References:
        [1] Paraschiv-Ionescu et al. (2019). Locomotion and cadence detection using a single trunk-fixed accelerometer ...

        [2] Paraschiv-Ionescu et al. (2020). Real-world speed estimation using single trunk IMU: methodological challenges ...

    """

    def __init__(
        self,
    ):
        """
        Initializes the ParaschivIonescuInitialContactDetection instance.
        """
        self.initial_contacts_ = None
        self.final_contacts_ = None

    def detect(
        self,
        accel_data: pd.DataFrame,
        sampling_freq_Hz: float,
        v_acc_col_name: str,
        gait_sequences: Optional[pd.DataFrame] = None,
        dt_data: Optional[pd.Series] = None,
        tracking_system: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detects initial contacts based on the input accelerometer data.

        Args:
            accel_data (pd.DataFrame): Input accelerometer data (N, 3) for x, y, and z axes.
            sampling_freq_Hz (float): Sampling frequency of the accelerometer data.
            v_acc_col_name (str): The column name that corresponds to the vertical acceleration.
            gait_sequences (pd.DataFrame, optional): A dataframe of detected gait sequences. If not provided, the entire acceleration time series will be used for detecting initial contacts.
            dt_data (pd.Series, optional): Original datetime in the input data. If original datetime is provided, the output onset will be based on that.
            tracking_system (str, optional): Tracking system the data is from to be used for events df. Default is None.

        Returns:
            ParaschivIonescuInitialContactDetection: Returns an instance of the class.
                The initial contacts information is stored in the 'initial_contacts_' attribute,
                which is a pandas DataFrame in BIDS format with the following columns:
                    - onset: Initial contacts.
                    - event_type: Type of the event (default is 'Inital contact').
                    - tracking_system: Tracking systems used the events are derived from.
        """
        # Check if data is empty
        if accel_data.empty:
            self.initial_contacts_ = pd.DataFrame()
            return self  # Return without performing further processing

        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (
            not isinstance(dt_data, pd.Series)
            or not pd.api.types.is_datetime64_any_dtype(dt_data)
        ):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if tracking_system is a string
        if tracking_system is not None and not isinstance(tracking_system, str):
            raise ValueError("tracking_system must be a string")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(accel_data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Convert acceleration data from "m/s^2" to "g"
        accel_data /= 9.81

        # Extract vertical accelerometer data using the specified index
        acc_vertical = accel_data[v_acc_col_name]
        self.acc_vertical = acc_vertical
        self.sampling_freq_Hz = sampling_freq_Hz

        # Initialize lists to store detected initial and final contacts
        initial_contacts_list = []
        final_contacts_list = []

        # The inforamtion of gait sequences will be used later for temporal_parameters 
        self.gait_sequences = gait_sequences

        # Ensure gait sequence is provided
        if gait_sequences is None:
            gait_sequences = pd.DataFrame(
              
                {"onset": [0], "duration": [len(accel_data) / sampling_freq_Hz]}
            )

        for _, gait_seq in gait_sequences.iterrows():
            # Calculate start and stop indices for the current gait sequence
            start_index = int(sampling_freq_Hz * gait_seq["onset"])
            stop_index = int(
                sampling_freq_Hz * (gait_seq["onset"] + gait_seq["duration"])
            )
            accv_gait_seq = acc_vertical[start_index:stop_index].to_numpy()

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs)
                initial_contacts_rel, final_contacts_rel = preprocessing.signal_decomposition_algorithm(
                    accv_gait_seq, sampling_freq_Hz
                )

                initial_contacts = gait_seq["onset"] + initial_contacts_rel
                final_contacts = gait_seq["onset"] + final_contacts_rel

                # Append detected contacts to the respective lists
                initial_contacts_list.extend(initial_contacts)
                final_contacts_list.extend(final_contacts)

            except Exception as e:
                print(
                    "Signal decomposition algorithm did not run successfully. Returning an empty vector of initial contacts"
                )
                print(f"Error: {e}")
                initial_contacts = []
                final_contacts = []

        # Check if any contacts were detected
        if not initial_contacts_list and not final_contacts_list:
            print("No initial or final contacts detected.")
            self.initial_contacts_ = pd.DataFrame()
            self.final_contacts_ = pd.DataFrame()
            return self
        # Create DataFrame for initial contacts
        self.initial_contacts_ = pd.DataFrame({
            "onset": initial_contacts_list,
            "event_type": ["initial contact"] * len(initial_contacts_list),
            "duration": 0,
            #"tracking_systems": tracking_system,
        })

        # Create DataFrame for final contacts
        self.final_contacts_ = pd.DataFrame({
            "onset": final_contacts_list,
            "event_type": ["final contact"] * len(final_contacts_list),
            "duration": 0,
            "tracking_systems": tracking_system,
        })

        # If original datetime is available, update the 'onset' column
        if dt_data is not None:
            for contacts_df in [self.initial_contacts_, self.final_contacts_]:
                valid_indices = [
                    index
                    for index in contacts_df["onset"]
                    if index < len(dt_data)
                ]
                invalid_indices = len(contacts_df["onset"]) - len(valid_indices)

                if invalid_indices > 0:
                    print(f"Warning: {invalid_indices} invalid index/indices found.")

                # Only use valid indices to access dt_data
                valid_dt_data = dt_data.iloc[valid_indices]

                # Update the 'onset' column
                contacts_df["onset"] = valid_dt_data.reset_index(drop=True)

        return self