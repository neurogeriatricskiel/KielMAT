# Import libraries
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from kielmat.utils import preprocessing
from kielmat.config import cfg_colors
from scipy.integrate import cumulative_trapezoid


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

    Additionally, spatial, temporophasic, temporal, and spatio-temporal parameters related to gait are calculated [1,3-6].

    Methods:
        detect(accel_data, gait_sequences, sampling_freq_Hz):
            Detects initial contacts on the accelerometer signal.

        temporal_parameters():
            Calculates the temporal parameters of the detected gaits using initial and final contacts information.
        
        temporophasic_parameters():
            Calculates the temporophasic_parameters parameters of the detected gaits using temporal parameters information.

         spatial_parameters():
            Calculates the spatial parameters of the detected gaits using temporal parameters and total walking distance. 

        spatio_temporal_parameters():
            Calculates the spatio-temporal parameters using temporal, spatial parameters and total walking distance.

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

        [3] McCamley et al. (2012) An enhanced estimate of initial contact and final contact instants of time ...

        [4] Hollman, John H., et al. Normative spatiotemporal gait parameters in older adults." Gait & posture ...

        [5] Hass, Chris J., et al. (2012). Quantitative normative gait data in a large cohort of ambulatory ...
        
        [6] Moe-Nilssen, Rolf, and Jorunn L. Helbostad (2020). Spatiotemporal gait parameters for older adults ...
    """

    def __init__(
        self,
    ):
        """
        Initializes the ParaschivIonescuInitialContactDetection instance.
        """
        self.initial_contacts_ = None
        self.final_contacts_ = None
        self.temporal_parameters_ = None
        self.temporophasic_parameters_ = None
        self.spatial_parameters_ = None
        self.spatio_temporal_parameters_ = None

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
            "tracking_systems": tracking_system,
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

    # Function to calculate temporal parameters
    def temporal_parameters(self):
        """
        Calculates temporal_parameters parameters using validated initial and final contact assignments.

        Returns:
            A DataFrame with temporal parameters, including:
                - step_time_l: Step times for the left foot (time between successive Initial Contacts of left and right feet).
                - step_time_r: Step times for the right foot (time between successive Initial Contacts of right and left feet).
                - stride_time_l: Stride times for the left foot (time between two successive Initial Contacts of the left foot).
                - stride_time_r: Stride times for the right foot (time between two successive Initial Contacts of the right foot).
                - swing_time_l: Swing times for the left foot (time between Initial Contact and Final Contact of the left foot).
                - swing_time_r: Swing times for the right foot (time between Initial Contact and Final Contact of the right foot).
                - stance_time_l: Stance times for the left foot (stride time minus swing time for the left foot).
                - stance_time_r: Stance times for the right foot (stride time minus swing time for the right foot).
                - single_support_time_l: Single support times for the left foot (time between Final Contact of the left foot and Initial Contact of the right foot).
                - single_support_time_r: Single support times for the right foot (time between Final Contact of the right foot and Initial Contact of the left foot).
                - double_support_time_l: Double support times for the left foot (time between Final Contact of the left foot and previous Initial Contact of the left foot).
                - double_support_time_r: Double support times for the right foot (time between Final Contact of the right foot and previous Initial Contact of the right foot).
                - cadence: Number of steps per minute.
        """
        if self.initial_contacts_ is None or self.final_contacts_ is None:
            raise ValueError("Initial and final contacts must be detected first.")

        all_parameters = []

        for seq_idx, gait_seq in self.gait_sequences.iterrows():
            start_time = gait_seq["onset"]
            stop_time = gait_seq["onset"] + gait_seq["duration"]

            # Filter initial and final contacts for this gait sequence
            gait_ic = self.initial_contacts_[
                (self.initial_contacts_["onset"] >= start_time) &
                (self.initial_contacts_["onset"] <= stop_time)
            ]["onset"].to_numpy()

            gait_fc = self.final_contacts_[
                (self.final_contacts_["onset"] >= start_time) &
                (self.final_contacts_["onset"] <= stop_time)
            ]["onset"].to_numpy()

            if len(gait_ic) < 2 or len(gait_fc) < 1:
                continue

            # Assign initial and final contact events to left and right feet
            ic_l, ic_r, fc_l, fc_r = self._assign_events_to_feet(gait_ic, gait_fc)

           # Step Time
            step_time_l = [
                ic_r[i] - ic_l[i]
                for i in range(min(len(ic_l), len(ic_r)))
            ]
            step_time_r = [
                ic_l[i + 1] - ic_r[i]
                for i in range(len(ic_r) - 1)
            ]

            # Stride Time
            stride_time_l = [ic_l[i + 1] - ic_l[i] for i in range(len(ic_l) - 1)]
            stride_time_r = [ic_r[i + 1] - ic_r[i] for i in range(len(ic_r) - 1)]

            # Swing Time
            swing_time_l = [
                fc_l[i] - ic_l[i]
                for i in range(len(fc_l) - 1)
            ]
            swing_time_r = [
                fc_r[i] - ic_r[i]
                for i in range(len(fc_r) - 1)
            ]

            # Stance Time: Calculated as stride time - swing time
            stance_time_l = [
                stride_time_l[i] - swing_time_l[i]
                for i in range(len(swing_time_l))
            ]
            stance_time_r = [
                stride_time_r[i] - swing_time_r[i]
                for i in range(len(swing_time_r))
            ]
 
            # Single Support Time
            single_support_time_l = [
                step_time_r[i] - swing_time_r[i]
                for i in range(min(len(step_time_r), len(swing_time_r)))
            ]
            single_support_time_r = [
                step_time_l[i] - swing_time_l[i]
                for i in range(min(len(step_time_l), len(swing_time_l)))
            ]

            # Double Support Time
            double_support_time_l = [
                stride_time_l[i] - (swing_time_l[i] + single_support_time_l[i])
                for i in range(min(len(stride_time_l), len(swing_time_l), len(single_support_time_l)))
            ]
            double_support_time_r = [
                stride_time_r[i] - (swing_time_r[i] + single_support_time_r[i])
                for i in range(min(len(stride_time_r), len(swing_time_r), len(single_support_time_r)))
            ]

            # Cadence
            total_time = gait_ic[-1] - gait_ic[0] if len(gait_ic) > 1 else None
            cadence = (len(gait_ic) / total_time) * 60 if total_time and total_time > 0 else np.nan

            # Append results
            all_parameters.append({
                "gait_sequence_id": seq_idx,
                "step_time_l": np.round(step_time_l, 3).tolist(),
                "step_time_r": np.round(step_time_r, 3).tolist(),
                "stride_time_l": np.round(stride_time_l, 3).tolist(),
                "stride_time_r": np.round(stride_time_r, 3).tolist(),
                "stance_time_l": np.round(stance_time_l, 3).tolist(),
                "stance_time_r": np.round(stance_time_r, 3).tolist(),
                "swing_time_l": np.round(swing_time_l, 3).tolist(),
                "swing_time_r": np.round(swing_time_r, 3).tolist(),
                "single_support_time_l": np.round(single_support_time_l, 3).tolist(),
                "single_support_time_r": np.round(single_support_time_r, 3).tolist(),
                "double_support_time_l": np.round(double_support_time_l, 3).tolist(),
                "double_support_time_r": np.round(double_support_time_r, 3).tolist(),
                "cadence": round(cadence, 2),
            })

        # Store results in a DataFrame
        self.temporal_parameters_ = pd.DataFrame(all_parameters)

        return self.temporal_parameters_

    # Function to calculate temporophasic parameters
    def temporophasic_parameters(self):
        """
        Calculates temporophasic parameters as percentages of the gait cycle.

        Returns:
            A DataFrame with temporophasic parameters, including:
                - stance_time_pct_gc_l: Stance time as % of the gait cycle for the left foot.
                - stance_time_pct_gc_r: Stance time as % of the gait cycle for the right foot.
                - swing_time_pct_gc_l: Swing time as % of the gait cycle for the left foot.
                - swing_time_pct_gc_r: Swing time as % of the gait cycle for the right foot.
                - single_support_time_pct_gc_l: Single support time as % of the gait cycle for the left foot.
                - single_support_time_pct_gc_r: Single support time as % of the gait cycle for the right foot.
                - double_support_time_pct_gc_l: Double support time as % of the gait cycle for the left foot.
                - double_support_time_pct_gc_r: Double support time as % of the gait cycle for the right foot.
        """
        if self.temporal_parameters_ is None:
            raise ValueError("Temporal parameters must be calculated first.")

        # Temporophasic parameters
        temporophasic_parameters_list = []

        for index, row in self.temporal_parameters_.iterrows():
            # Handle mismatched lengths gracefully
            min_stride_l = len(row["stride_time_l"])
            min_stride_r = len(row["stride_time_r"])

            stance_time_pct_gc_l = [
                (row["stance_time_l"][i] / row["stride_time_l"][i]) * 100
                if i < min_stride_l and row["stride_time_l"][i] > 0 else np.nan
                for i in range(len(row["stance_time_l"]))
            ]
            stance_time_pct_gc_r = [
                (row["stance_time_r"][i] / row["stride_time_r"][i]) * 100
                if i < min_stride_r and row["stride_time_r"][i] > 0 else np.nan
                for i in range(len(row["stance_time_r"]))
            ]
            swing_time_pct_gc_l = [
                (row["swing_time_l"][i] / row["stride_time_l"][i]) * 100
                if i < min_stride_l and row["stride_time_l"][i] > 0 else np.nan
                for i in range(len(row["swing_time_l"]))
            ]
            swing_time_pct_gc_r = [
                (row["swing_time_r"][i] / row["stride_time_r"][i]) * 100
                if i < min_stride_r and row["stride_time_r"][i] > 0 else np.nan
                for i in range(len(row["swing_time_r"]))
            ]
            single_support_time_pct_gc_l = [
                (row["single_support_time_l"][i] / row["stride_time_l"][i]) * 100
                if i < min_stride_l and row["stride_time_l"][i] > 0 else np.nan
                for i in range(len(row["single_support_time_l"]))
            ]
            single_support_time_pct_gc_r = [
                (row["single_support_time_r"][i] / row["stride_time_r"][i]) * 100
                if i < min_stride_r and row["stride_time_r"][i] > 0 else np.nan
                for i in range(len(row["single_support_time_r"]))
            ]
            double_support_time_pct_gc_l = [
                (row["double_support_time_l"][i] / row["stride_time_l"][i]) * 100
                if i < min_stride_l and row["stride_time_l"][i] > 0 else np.nan
                for i in range(len(row["double_support_time_l"]))
            ]
            double_support_time_pct_gc_r = [
                (row["double_support_time_r"][i] / row["stride_time_r"][i]) * 100
                if i < min_stride_r and row["stride_time_r"][i] > 0 else np.nan
                for i in range(len(row["double_support_time_r"]))
            ]

            # Append results
            temporophasic_parameters_list.append({
                "gait_sequence_id": row["gait_sequence_id"],
                "stance_time_pct_gc_l": np.round(stance_time_pct_gc_l, 2).tolist(),
                "stance_time_pct_gc_r": np.round(stance_time_pct_gc_r, 2).tolist(),
                "swing_time_pct_gc_l": np.round(swing_time_pct_gc_l, 2).tolist(),
                "swing_time_pct_gc_r": np.round(swing_time_pct_gc_r, 2).tolist(),
                "single_support_time_pct_gc_l": np.round(single_support_time_pct_gc_l, 2).tolist(),
                "single_support_time_pct_gc_r": np.round(single_support_time_pct_gc_r, 2).tolist(),
                "double_support_time_pct_gc_l": np.round(double_support_time_pct_gc_l, 2).tolist(),
                "double_support_time_pct_gc_r": np.round(double_support_time_pct_gc_r, 2).tolist(),
            })

        # Store results in a DataFrame
        self.temporophasic_parameters_ = pd.DataFrame(temporophasic_parameters_list)

        return self.temporophasic_parameters_

    # Function to calculate spatial parameters
    def spatial_parameters(
        self,
        wearable_height: float = 1.0,  # Default height in meters
    ):
        """
        Calculates spatial parameters of the detected gaits.

        Args:
            wearable_height (float): Height of the wearable device from the ground, in meters. Default is 1.0.

        Returns:
            pd.DataFrame: A DataFrame with spatial parameters, including:
                - step_length_l: Lengths of steps for the left foot.
                - step_length_r: Lengths of steps for the right foot.
                - stride_length_l: Stride lengths for the left foot.
                - stride_length_r: Stride lengths for the right foot.
        """
        if self.initial_contacts_ is None or self.final_contacts_ is None:
            raise ValueError("Initial and final contacts must be detected first.")

        spatial_params_list = []

        for seq_idx, gait_seq in self.gait_sequences.iterrows():
            start_time = gait_seq["onset"]
            stop_time = gait_seq["onset"] + gait_seq["duration"]

            # Filter initial and final contacts for this gait sequence
            gait_ic = self.initial_contacts_[
                (self.initial_contacts_["onset"] >= start_time) &
                (self.initial_contacts_["onset"] <= stop_time)
            ]["onset"].to_numpy()

            gait_fc = self.final_contacts_[
                (self.final_contacts_["onset"] >= start_time) &
                (self.final_contacts_["onset"] <= stop_time)
            ]["onset"].to_numpy()

            if len(gait_ic) < 2 or len(gait_fc) < 1:
                continue

            # Assign initial and final contact events to left and right feet
            ic_l, ic_r, fc_l, fc_r = self._assign_events_to_feet(gait_ic, gait_fc)

            # Calculate step lengths using vertical acceleration
            step_length_l = []
            step_length_r = []

            for i in range(len(fc_l)):
                # Left step length
                if i < len(ic_l) and i < len(fc_l):
                    acc_segment_l = self.acc_vertical[
                        int(self.sampling_freq_Hz * ic_l[i]):int(self.sampling_freq_Hz * fc_l[i])
                    ]
                    vertical_displacement_l = max(cumulative_trapezoid(acc_segment_l, dx=1/self.sampling_freq_Hz, initial=0)) - \
                                            min(cumulative_trapezoid(acc_segment_l, dx=1/self.sampling_freq_Hz, initial=0))
                    step_length_l.append(2 * np.sqrt(2 * wearable_height * vertical_displacement_l - vertical_displacement_l**2))

                # Right step length
                if i < len(ic_r) and i < len(fc_r):
                    acc_segment_r = self.acc_vertical[
                        int(self.sampling_freq_Hz * ic_r[i]):int(self.sampling_freq_Hz * fc_r[i])
                    ]
                    vertical_displacement_r = max(cumulative_trapezoid(acc_segment_r, dx=1/self.sampling_freq_Hz, initial=0)) - \
                                            min(cumulative_trapezoid(acc_segment_r, dx=1/self.sampling_freq_Hz, initial=0))
                    step_length_r.append(2 * np.sqrt(2 * wearable_height * vertical_displacement_r - vertical_displacement_r**2))

            # Calculate stride lengths
            stride_length_l = [
                step_length_l[i] + step_length_r[i]
                for i in range(len(step_length_l) - 1)
            ]
            stride_length_r = [
                step_length_r[i] + step_length_l[i + 1]
                for i in range(len(step_length_r) - 1)
            ]

            # Append results
            spatial_params_list.append({
                "gait_sequence_id": seq_idx,
                "step_length_l": np.round(step_length_l, 3).tolist(),
                "step_length_r": np.round(step_length_r, 3).tolist(),
                "stride_length_l": np.round(stride_length_l, 3).tolist(),
                "stride_length_r": np.round(stride_length_r, 3).tolist(),
            })

        # Store results in a DataFrame
        self.spatial_parameters_ = pd.DataFrame(spatial_params_list)

        return self.spatial_parameters_

    # Function to calculate spatio-temporal parameters
    def spatio_temporal_parameters(self):
        """
        Calculates spatio-temporal parameters, including gait speed and stride speed, 
        using stride lengths and stride times.

        Returns:
            A DataFrame with spatio-temporal parameters, including:
                - gait_speed: Average gait speed (m/s).
                - stride_speed_l: Stride speed for the left foot (m/s).
                - stride_speed_r: Stride speed for the right foot (m/s).
        """
        if self.spatial_parameters_ is None:
            raise ValueError("Spatial parameters must be calculated first using `spatial_parameters` method.")

        if self.temporal_parameters_ is None:
            raise ValueError("Temporal parameters must be calculated first using `temporal_parameters` method.")

        spatio_temporal_params_list = []

        for spatial_row, temporal_row in zip(self.spatial_parameters_.itertuples(), self.temporal_parameters_.itertuples()):
            # Calculate stride speeds
            stride_speed_l = [
                stride_length / stride_time if stride_time > 0 else np.nan
                for stride_length, stride_time in zip(spatial_row.stride_length_l, temporal_row.stride_time_l)
            ]
            stride_speed_r = [
                stride_length / stride_time if stride_time > 0 else np.nan
                for stride_length, stride_time in zip(spatial_row.stride_length_r, temporal_row.stride_time_r)
            ]

            # Calculate gait speed as the average of stride speeds
            all_stride_lengths = spatial_row.stride_length_l + spatial_row.stride_length_r
            all_stride_times = temporal_row.stride_time_l + temporal_row.stride_time_r
            if len(all_stride_lengths) > 0 and len(all_stride_times) > 0:
                gait_speed = np.sum(all_stride_lengths) / np.sum(all_stride_times)
            else:
                gait_speed = np.nan

            # Append the results
            spatio_temporal_params_list.append({
                "gait_sequence_id": spatial_row.gait_sequence_id,
                "gait_speed": round(gait_speed, 3),
                "stride_speed_l": np.round(stride_speed_l, 3).tolist(),
                "stride_speed_r": np.round(stride_speed_r, 3).tolist(),
            })

        # Store the results in a DataFrame
        self.spatio_temporal_parameters_ = pd.DataFrame(spatio_temporal_params_list)

        return self.spatio_temporal_parameters_

    # Function to assign left and right feet
    def _assign_events_to_feet(self, gait_ic, gait_fc):
        """
        Distinguish left and right foot contacts.

        Args:
            gait_ic (np.ndarray): Array of initial contacts (ICs).
            gait_fc (np.ndarray): Array of final contacts (FCs).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays of IC and FC for left and right feet.
        """
        ic_l, ic_r, fc_l, fc_r = [], [], [], []

        # Strict alternation of foot assignment
        last_foot = None
        for current_onset in gait_ic:
            if last_foot == "left":
                ic_r.append(current_onset)
                last_foot = "right"
            else:
                ic_l.append(current_onset)
                last_foot = "left"

        # Assign final contacts based on nearest preceding initial contact
        for fc_onset in gait_fc:
            preceding_ic_l = [ic for ic in ic_l if ic <= fc_onset]
            preceding_ic_r = [ic for ic in ic_r if ic <= fc_onset]

            if preceding_ic_l and (not preceding_ic_r or preceding_ic_l[-1] > preceding_ic_r[-1]):
                fc_l.append(fc_onset)
            elif preceding_ic_r:
                fc_r.append(fc_onset)

        return np.array(ic_l), np.array(ic_r), np.array(fc_l), np.array(fc_r)