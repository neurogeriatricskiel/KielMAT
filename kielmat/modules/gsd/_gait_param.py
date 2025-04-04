import numpy as np
import pandas as pd
from kielmat.utils import preprocessing
from scipy.integrate import cumulative_trapezoid


class GaitSpatioTemporalParameters:
    """
    The `GaitSpatioTemporalParameters` class computes spatio-temporal gait parameters from 
    pre-detected gait events such as initial contacts (IC), final contacts (FC), and gait sequences.
    It is designed for analyzing human gait using IMU-derived temporal event data.

    Parameters are computed based on clinically validated definitions and peer-reviewed literature.
    Inputs must be structured as pandas DataFrames, with event types labeled and time values in seconds.

    Key Features:

    - **Temporal parameters** (step time [s], stride time [s], stance time [s], swing time [s], 
      single and double support time [s])  
      → Based on [2, 3, 4].

    - **Temporophasic parameters** (stance, swing, single and double support time as % of stride)  
      → Based on [2, 3].

    - **Spatial parameters** (step and stride lengths [m] using the inverted pendulum model)  
      → Based on [5].

    - **Spatio-temporal parameters** (stride speed [m/s] for each leg)  
      → Based on [2–4].

    Notes:

    - Event inputs must follow a BIDS-like schema with columns: `onset`, `duration`, `event_type`, `rl_label`, and `tracking_system`.
    - All timestamps should be in seconds.
    - Left and right steps are handled separately; incomplete strides are skipped.

    Methods:
        detect(gait_sequences, initial_contacts, final_contacts):
            Loads pre-detected gait events into the class.

        temporal_parameters():
            Computes step and stride timing parameters.

        temporophasic_parameters():
            Computes stride phase durations as percentages.

        spatial_parameters(accel_data, v_acc_col_name, sampling_freq_Hz, wearable_height=1.0):
            Computes step and stride lengths using vertical acceleration.

        spatiotemporal_parameters():
            Computes stride speed using stride time and length.


    Examples:
        >>> gait_stp = GaitSpatioTemporalParameters()
        >>> gait_stp.detect(gait_sequences, initial_contacts, final_contacts)

        >>> gait_stp.temporal_parameters()
        >>> print(gait_stp.step_temporal_parameters_)       # Step timing [s]
        >>> print(gait_stp.stride_temporal_parameters_)     # Stride timing [s]

        >>> gait_stp.temporophasic_parameters()
        >>> print(gait_stp.temporophasic_parameters_)       # [% stride]

        >>> gait_stp.spatial_parameters(
                accel_data=accel_df,
                v_acc_col_name="pelvis_ACCEL_z",
                sampling_freq_Hz=100,
                wearable_height=1.0
        )
        >>> print(gait_stp.step_spatial_parameters_)        # Step length [m]
        >>> print(gait_stp.stride_spatial_parameters_)      # Stride length [m]

        >>> gait_stp.spatiotemporal_parameters()
        >>> print(gait_stp.spatiotemporal_parameters_)      # Stride speed [m/s]

    References:

    [1] Zijlstra, W., & Hof, A. L. (2003). Assessment of spatio-temporal gait parameters from trunk accelerations during human walking. *Gait & Posture*, 18(2), 1–10.
    
    [2] Moe-Nilssen, R., Helbostad, J. L., et al. (2020). Spatiotemporal gait parameters for older adults. *Gait & Posture*, 80, 63–69.
    
    [3] Hollman, J. H., et al. (2011). Normative spatiotemporal gait parameters in older adults. *Gait & Posture*, 34(1), 111–118.
    
    [4] Hass, C. J., et al. (2012). Quantitative normative gait data in a large cohort of ambulatory persons with Parkinson’s disease. *PLOS ONE*, 7(8), e42337.
    
    [5] Cerny, M., Noury, N., & Deplorte, L. (2015). Validation of the inverted pendulum model for gait length calculation. *EMBC 2015 - IEEE Engineering in Medicine and Biology Conference*.
    """

    def __init__(
        self,
     
    ):
        """
        Initializes the GaitSpatioTemporalParameters instance.
        """
        self.temporal_parameters_ = None
        self.temporophasic_parameters_ = None
        self.spatial_parameters_ = None
        self.spatiotemporal_parameters_ = None

    def detect(
        self,
        gait_sequences: pd.DataFrame,
        initial_contacts: pd.DataFrame,
        final_contacts: pd.DataFrame,
    ):
        """
        Loads the event information to be used for spatio-temporal calculations.

        Args:
            gait_sequences (pd.DataFrame): Gait sequence events with 'onset' and 'duration'.
            initial_contacts (pd.DataFrame): Initial contacts with 'onset' and 'rl_label'.
            final_contacts (pd.DataFrame): Final contacts with 'onset' and 'rl_label'.

        Returns:
            GaitSpatioTemporalParameters: Returns the instance with updated attributes.
        """
        self.gait_sequences = gait_sequences
        self.initial_contacts = initial_contacts
        self.final_contacts = final_contacts

        return self


    # Function to calculate gait temporal parameters
    def temporal_parameters(self) -> pd.DataFrame:
        """
        Calculates temporal gait parameters from detected initial and final contact events,
        and stores the result as a step and stride parameter tables.

        Returns:
            The step temporal parameteres are stored in the 'step_temporal_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id: Index of the gait sequence.
                - step_id: Sequential ID of the step within the sequence.
                - foot: The initiating foot (left or right).
                - onset: Time of the step's initial contact (s).
                - end_time: Time of the next IC (s).
                - step_time: Duration between onset and end_time (s).

            The stride temporal parameteres are stored in the 'stride_temporal_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id: Index of the gait sequence.
                - stride_id: Sequential ID of the stride within the sequence.
                - foot: The initiating foot (left or right).
                - onset: Time of the stride's initial contact (s).
                - end_time: Time of the next IC of the same foot (s).
                - stride_time: Duration between onset and end_time (s).
                - stance_time: Duration of stance phase (s).
                - swing_time: Duration of swing phase (s).
                - single_support_time: Duration with only one foot on the ground (s).
                - double_support_time: Duration with both feet on the ground (s).

        Notes:
            - Steps alternate between feet (left → right or right → left), while strides are defined as
            consecutive initial contacts of the same foot.
        """
        # Initialize list to store step data for all sequences
        step_rows = []

        # Initialize list to store stride data for all sequences
        stride_rows = []

        # Loop through each gait sequence
        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]                        # Start time of gait sequence
            end = seq["onset"] + seq["duration"]        # End time of gait sequence

            # Select initial contacts within current sequence window
            ic = self.initial_contacts[
                (self.initial_contacts["onset"] >= start) &
                (self.initial_contacts["onset"] <= end)
            ]

            # Select final contacts within current sequence window
            fc = self.final_contacts[
                (self.final_contacts["onset"] >= start) &
                (self.final_contacts["onset"] <= end)
            ]

            # Skip sequence if there are not enough ICs or FCs to compute parameters
            if len(ic) < 2 or len(fc) < 1:
                continue

            # Sort ICs by onset time and convert to list of (onset, foot) tuples
            ic_sorted = sorted(
                ic[["onset", "rl_label"]].to_records(index=False),
                key=lambda x: x[0]
            )

            # CALCULATION OF STEP PARAMETERS
            step_counter = 0  # Initialize step ID counter for this sequence

            # Loop through consecutive ICs
            for i in range(len(ic_sorted) - 1):
                t1, lab1 = ic_sorted[i]       # Current IC: time and foot
                t2, lab2 = ic_sorted[i + 1]   # Next IC: time and foot

                # Only count steps if feet alternate (e.g., left to right)
                if lab1 != lab2:
                    step_time = round(t2 - t1, 3)  # Calculate step duration
                    step_rows.append({             # Store step row
                        "gait_sequence_id": seq_idx,
                        "step_id": step_counter,
                        "foot": lab1,
                        "onset": t1,
                        "end_time": t2,
                        "step_time": step_time
                    })
                    step_counter += 1  # Increment step ID

            # CALCULATION OF STRIDE PARAMETERS
            temp_stride_rows = []  # Temporary list for this sequence's strides

            # Loop over each foot separately
            for foot in ["left", "right"]:
                # Get IC and FC for this foot
                ic_foot = np.sort(ic[ic["rl_label"] == foot]["onset"].to_numpy())
                fc_foot = np.sort(fc[fc["rl_label"] == foot]["onset"].to_numpy())

                # Get IC and FC for the opposite foot
                other_foot = "right" if foot == "left" else "left"
                ic_other = np.sort(ic[ic["rl_label"] == other_foot]["onset"].to_numpy())
                fc_other = np.sort(fc[fc["rl_label"] == other_foot]["onset"].to_numpy())

                # Loop through consecutive ICs of the same foot (strides)
                for i in range(len(ic_foot) - 1):
                    t_ic1 = ic_foot[i]            # Current stride start time
                    t_ic2 = ic_foot[i + 1]        # Current stride end time

                    stride_time = round(t_ic2 - t_ic1, 3)  # Duration of stride

                    # Find FC within the stride to compute stance time
                    fc_in_range = fc_foot[(fc_foot > t_ic1) & (fc_foot < t_ic2)]
                    stance_time = round(fc_in_range[0] - t_ic1, 3) if fc_in_range.size > 0 else np.nan

                    # Swing = stride - stance
                    swing_time = round(stride_time - stance_time, 3) if not np.isnan(stance_time) else np.nan

                    # Support phase calculations using both feet
                    ic_other_in = ic_other[(ic_other > t_ic1) & (ic_other < t_ic2)]
                    fc_other_in = fc_other[(fc_other > t_ic1) & (fc_other < t_ic2)]
                    fc_this_in = fc_foot[(fc_foot > t_ic1) & (fc_foot < t_ic2)]

                    if len(ic_other_in) >= 1 and len(fc_other_in) >= 1 and len(fc_this_in) >= 1:
                        ds1 = fc_other_in[0] - t_ic1                 # First double support phase
                        ds2 = fc_this_in[0] - ic_other_in[0]         # Second double support phase
                        double_support = round(ds1 + ds2, 3)         # Total double support
                        single_support = round(stance_time - double_support, 3) if not np.isnan(stance_time) else np.nan
                    else:
                        double_support = np.nan
                        single_support = np.nan

                    # Append stride row
                    temp_stride_rows.append({
                        "gait_sequence_id": seq_idx,
                        "foot": foot,
                        "onset": t_ic1,
                        "end_time": t_ic2,
                        "stride_time": stride_time,
                        "stance_time": stance_time,
                        "swing_time": swing_time,
                        "single_support_time": single_support,
                        "double_support_time": double_support
                    })

            # Sort stride rows by onset and assign stride_id
            temp_stride_rows = sorted(temp_stride_rows, key=lambda x: x["onset"])
            for idx, row in enumerate(temp_stride_rows):
                row["stride_id"] = idx         # Assign stride ID based on order
                stride_rows.append(row)        # Add row to global stride list

        # Create DataFrame from step rows
        self.step_temporal_parameters_ = pd.DataFrame(step_rows)[
            ["gait_sequence_id", "step_id", "foot", "onset", "end_time", "step_time"]
        ]

        # Create DataFrame from stride ROWS
        self.stride_temporal_parameters_ = pd.DataFrame(stride_rows)[
            [
                "gait_sequence_id", "stride_id", "foot", "onset", "end_time",
                "stride_time", "stance_time", "swing_time",
                "single_support_time", "double_support_time"
            ]
        ]

        # Return the instance for chaining if desired
        return self


    # Function to calculate gait temporophasic parameters
    def temporophasic_parameters(self) -> pd.DataFrame:
        """
        Calculates temporophasic parameters as percentage of the gait cycle from stride parameters.

        Returns:
            The temporophasic parameteres are stored in the 'temporophasic_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id: Index of the gait sequence.
                - stride_id: ID of the stride
                - foot: Left or right.
                - stance_pct: Stance time as % of stride duration.
                - swing_pct: Swing time as % of stride duration.
                - single_support_pct: Time with one foot in contact (% of stride duration).
                - double_support_pct: Time with both feet in contact (% of stride duration).

        Notes:
            - All percentage values are rounded to two decimal places.
            - Requires `temporal_parameters()` to be called first.
        """
        # Ensure stride parameters are available
        if self.stride_temporal_parameters_ is None:
            raise ValueError("Call temporal_parameters() before computing temporophasic_parameters().")

        # Initialize output list
        tempophasic_rows = []

        # Iterate over each stride
        for _, row in self.stride_temporal_parameters_.iterrows():
            stride_time = row["stride_time"]               # Total stride time
            stance_time = row["stance_time"]               # Stance duration
            swing_time = row["swing_time"]                 # Swing duration
            ss_time = row["single_support_time"]           # Single support time
            ds_time = row["double_support_time"]           # Double support time

            # Calculate temporophasic percentages
            stance_pct = round((stance_time / stride_time) * 100, 2) if stride_time > 0 else np.nan
            swing_pct = round((swing_time / stride_time) * 100, 2) if stride_time > 0 else np.nan
            single_pct = round((ss_time / stride_time) * 100, 2) if stride_time > 0 and not np.isnan(ss_time) else np.nan
            double_pct = round((ds_time / stride_time) * 100, 2) if stride_time > 0 and not np.isnan(ds_time) else np.nan

            # Append the result as one row per stride
            tempophasic_rows.append({
                "gait_sequence_id": row["gait_sequence_id"],
                "stride_id": row["stride_id"],
                "foot": row["foot"],
                "stance_pct": stance_pct,
                "swing_pct": swing_pct,
                "single_support_pct": single_pct,
                "double_support_pct": double_pct
            })

        # Store as DataFrame in the instance
        self.temporophasic_parameters_ = pd.DataFrame(tempophasic_rows)[
            [
                "gait_sequence_id", "stride_id", "foot",
                "stance_pct", "swing_pct", "single_support_pct", "double_support_pct"
            ]
        ]

        # Return self for chaining
        return self


    # Function to calculate gait spatial parameters
    def spatial_parameters(
        self,
        accel_data: pd.DataFrame,
        v_acc_col_name: str,
        sampling_freq_Hz: float,
        wearable_height: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculates spatial gait parameters using vertical acceleration data and detected gait events.
        Outputs step and stride lengths using the inverted pendulum model.

        Returns:
            The step spatial parameteres are stored in the 'step_spatial_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id: Index of gait sequence
                - step_id: Step number within sequence
                - foot: Leading foot (left or right)
                - step_length: Estimated step length in meters

            The step spatial parameteres are stored in the 'stride_spatial_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id: Index of gait sequence
                - stride_id: Stride number within sequence
                - foot: Foot completing the stride
                - stride_length: Sum of two consecutive steps (meters)
        """
        # Raise error if required temporal step data is missing
        if self.gait_sequences is None or self.step_temporal_parameters_ is None:
            raise ValueError("Call detect() and temporal_parameters() before spatial_parameters().")

        # Create lists to store individual step and stride results
        step_spatial_rows = []
        stride_spatial_rows = []

        # Extract vertical acceleration signal as a NumPy array
        acc_vertical = accel_data[v_acc_col_name].to_numpy().copy()

        # Remove mean from signal to remove bias
        acc_vertical -= np.mean(acc_vertical)

        # Apply low-pass filter to smooth the vertical acceleration
        acc_vertical = preprocessing.lowpass_filter(acc_vertical, method="butter", order=4, fs=sampling_freq_Hz)

        # Loop over each gait sequence window
        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]  # Start time of current gait sequence
            end = start + seq["duration"]  # End time of current gait sequence

            # Get all ICs within this sequence
            ic_seq = self.initial_contacts[
                (self.initial_contacts["onset"] >= start) &
                (self.initial_contacts["onset"] <= end)
            ]

            # Skip if fewer than two ICs (can't compute even one step)
            if len(ic_seq) < 2:
                continue

            # Sort initial contacts by time (onset) and convert to (onset, foot) tuples
            ic_sorted = sorted(ic_seq[["onset", "rl_label"]].to_records(index=False), key=lambda x: x[0])

            step_events = []  # Store step direction and length for stride construction
            step_counter = 0  # Initialize counter for assigning step_id

            # Loop through each pair of consecutive ICs
            for i in range(len(ic_sorted) - 1):
                t1, lab1 = ic_sorted[i]         # First event: time and foot label
                t2, lab2 = ic_sorted[i + 1]     # Second event: time and foot label

                # Only process alternating feet (i.e., valid step)
                if lab1 != lab2:
                    idx_start = int(t1 * sampling_freq_Hz)  # Convert time to sample index
                    idx_end = int(t2 * sampling_freq_Hz)    # Convert next time to index

                    # Ensure indices are valid and segment is within signal range
                    if idx_end > idx_start and idx_end <= len(acc_vertical):
                        acc_seg = acc_vertical[idx_start:idx_end]  # Extract signal segment between two ICs

                        vel = cumulative_trapezoid(acc_seg, dx=1 / sampling_freq_Hz, initial=0)  # Integrate to get velocity
                        disp = cumulative_trapezoid(vel, dx=1 / sampling_freq_Hz, initial=0)     # Integrate again for displacement

                        delta_z = np.max(disp) - np.min(disp)  # Vertical displacement during step

                        try:
                            # Compute step length using inverted pendulum model
                            step_length = 2 * np.sqrt(2 * wearable_height * delta_z - delta_z ** 2)
                        except Exception:
                            step_length = np.nan  # If invalid (e.g., negative sqrt), assign NaN

                        # Append step result
                        step_spatial_rows.append({
                            "gait_sequence_id": seq_idx,
                            "step_id": step_counter,
                            "foot": lab1,
                            "step_length": round(step_length, 3)
                        })

                        # Save for stride building
                        step_events.append((lab1, step_length))
                        step_counter += 1  # Increment step ID

            # Construct strides from consecutive valid steps
            for i in range(len(step_events) - 1):
                f1, l1 = step_events[i]         # First step: foot and length
                f2, l2 = step_events[i + 1]     # Second step: foot and length

                # Only pair opposite foot steps into a stride
                if (f1 == "left" and f2 == "right") or (f1 == "right" and f2 == "left"):
                    stride_length = round(l1 + l2, 3)  # Add two step lengths
                    stride_foot = f1  # Assign stride to foot initiating it

                    stride_spatial_rows.append({
                        "gait_sequence_id": seq_idx,
                        "foot": stride_foot,
                        "stride_length": stride_length
                    })

        # Convert step results to DataFrame
        self.step_spatial_parameters_ = pd.DataFrame(step_spatial_rows)[
            ["gait_sequence_id", "step_id", "foot", "step_length"]
        ]

        # Convert stride results to DataFrame and assign stride_id
        df_stride_spatial = pd.DataFrame(stride_spatial_rows)  # Create stride DataFrame
        df_stride_spatial = df_stride_spatial.sort_values(by=["gait_sequence_id"]).reset_index(drop=True)  # Sort by sequence
        df_stride_spatial["stride_id"] = df_stride_spatial.groupby("gait_sequence_id").cumcount()  # Assign stride_id per sequence

        # Store as class attribute
        self.stride_spatial_parameters_ = df_stride_spatial[
            ["gait_sequence_id", "stride_id", "foot", "stride_length"]
        ]

        # Return self for chaining
        return self

    def spatiotemporal_parameters(self) -> pd.DataFrame:
        """
        Calculates stride-level spatio-temporal gait parameters by combining stride times
        and stride lengths from previously computed temporal and spatial parameters.

        Returns:
            The stride spatio-temporal parameteres are stored in the 'spatiotemporal_parameters_' attribute, which is a pandas DataFrame with the following information:

                - gait_sequence_id
                - stride_id
                - foot
                - stride_speed (m/s)
        """
        # Ensure required temporal and spatial stride parameters exist
        if self.stride_temporal_parameters_ is None or self.stride_spatial_parameters_ is None:
            raise ValueError("Call temporal_parameters() and spatial_parameters() before spatiotemporal_parameters().")

        # Merge temporal and spatial stride-level parameters
        merged = pd.merge(
            self.stride_temporal_parameters_,
            self.stride_spatial_parameters_,
            on=["gait_sequence_id", "stride_id", "foot"],
            how="inner"  # keep only matching strides
        )

        # Compute stride speed: stride length / stride time
        merged["stride_speed"] = merged["stride_length"] / merged["stride_time"]
        merged["stride_speed"] = merged["stride_speed"].round(3)

        # Select final output columns
        self.spatiotemporal_parameters_ = merged[[
            "gait_sequence_id", "stride_id", "foot", "stride_speed"
        ]]

        return self


