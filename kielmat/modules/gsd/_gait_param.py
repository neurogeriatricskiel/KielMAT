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

    - **Temporal parameters** (step time, stride time, stance time, swing time, cadence, single and double support times)  
    → Based on [2, 3, 4].

    - **Temporophasic parameters** (stance, swing, single and double support time as % of stride)  
    → Derived following conventions in [2, 3].

    - **Spatial parameters** (step and stride lengths using the inverted pendulum model and vertical displacement)  
    → Based on the method by Cerny et al. [5].

    - **Spatio-temporal parameters** (gait speed and stride speed for left and right legs)  
    → Combines outputs from [2–4].

    Notes:

    - Event inputs must follow a BIDS-like schema with columns: `onset`, `duration`, `event_type`, `rl_label`, and `tracking_system`.
    - All timestamps should be in seconds.
    - Left and right steps are handled separately; incomplete strides are skipped.

    Methods:

    - `detect(gait_sequences, initial_contacts, final_contacts)`
    - `temporal_parameters()`
    - `temporophasic_parameters()`
    - `spatial_parameters(accel_data, v_acc_col_name, sampling_freq_Hz, wearable_height=1.0)`
    - `spatiotemporal_parameters()`

    Examples:

        >>> gait_stp = GaitSpatioTemporalParameters()
        >>> gait_stp.detect(gait_sequences, initial_contacts, final_contacts)

        >>> temporal_df = gait_stp.temporal_parameters()
        >>> print(temporal_df.temporal_parameters_)

        >>> temporophasic_df = gait_stp.temporophasic_parameters()
        >>> print(temporophasic_df.temporophasic_parameters_)

        >>> spatial_df = gait_stp.spatial_parameters(
                accel_data=accel_df, 
                v_acc_col_name="pelvis_ACCEL_z", 
                sampling_freq_Hz=100, 
                wearable_height=1.0
         )
        >>> print(spatial_df.spatial_parameters_)

        >>> spatiotemporal_df = gait_stp.spatiotemporal_parameters()
        >>> print(spatiotemporal_df.spatiotemporal_parameters_)

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
        Calculates temporal gait parameters for each gait sequence:

        Returns:
            A DataFrame with temporal parameters, including:

                - step_time_l / step_time_r: Time from an initial contact (IC) of one foot to the next IC of the opposite foot.
                - stride_time_l / stride_time_r: Time between consecutive ICs of the same foot.
                - stance_time_l / stance_time_r: Time from an IC to the first final contact (FC) in the same stride.
                - swing_time_l / swing_time_r: Stride time minus stance time.
                - single_support_time_l / single_support_time_r: Time during stance when only one foot contacts the ground.
                - double_support_time_l / double_support_time_r: Time when both feet are on the ground during gait.
                - cadence: Total number of steps per minute.
        """
        # Initialize an empty list to store the gait temporal parameters for each gait sequence
        temporal_parameters_df = []  

        # Loop through each gait sequence by index and row
        for seq_idx, seq in self.gait_sequences.iterrows():
            # Determine the start and end times for the current gait sequence
            start = seq["onset"]
            end = seq["onset"] + seq["duration"]

            # Select all initial contact (IC) events that occur within the current gait sequence time window
            ic = self.initial_contacts[
                (self.initial_contacts["onset"] >= start) & 
                (self.initial_contacts["onset"] <= end)
            ]
            # Select all final contact (FC) events that occur within the current gait sequence time window
            fc = self.final_contacts[
                (self.final_contacts["onset"] >= start) & 
                (self.final_contacts["onset"] <= end)
            ]
            # Skip this gait sequence if there are not enough IC or FC events
            if len(ic) < 2 or len(fc) < 1:
                continue

            # Retrieve the first IC and FC events to check for any timing adjustment
            first_ic_time = ic.iloc[0]["onset"]
            first_ic_label = ic.iloc[0]["rl_label"]
            first_fc_time = fc.iloc[0]["onset"]
            first_fc_label = fc.iloc[0]["rl_label"]

            # If the first FC occurs before the first IC for the same foot, drop that FC event
            if first_fc_time < first_ic_time and first_fc_label == first_ic_label:
                fc = fc.iloc[1:]

            # Create a sorted list of IC events (time and label) based on onset time
            ic_sorted = sorted(
                ic[["onset", "rl_label"]].to_records(index=False),
                key=lambda x: x[0]
            )

            # Initialize lists to hold step times for alternating transitions:
            # step_time_r for right-to-left (R→L) transitions and step_time_l for left-to-right (L→R) transitions
            step_time_r = []  # R→L transitions (from a right IC to the next left IC)
            step_time_l = []  # L→R transitions (from a left IC to the next right IC)

            # Loop over consecutive pairs of sorted IC events
            for i in range(len(ic_sorted) - 1):
                t1, lab1 = ic_sorted[i]      # Current event's time and label
                t2, lab2 = ic_sorted[i+1]    # Next event's time and label

                # If the current event is from the right and the next from the left, record as R→L
                if lab1 == "right" and lab2 == "left":
                    step_time_r.append(round(t2 - t1, 3))

                # If the current event is from the left and the next from the right, record as L→R
                elif lab1 == "left" and lab2 == "right":
                    step_time_l.append(round(t2 - t1, 3))

            # For stride times, compute the time differences between consecutive IC events for each foot separately.
            ic_left = np.sort(ic[ic["rl_label"] == "left"]["onset"].to_numpy())
            ic_right = np.sort(ic[ic["rl_label"] == "right"]["onset"].to_numpy())
            stride_time_left = [round(ic_left[i+1] - ic_left[i], 3) for i in range(len(ic_left) - 1)]
            stride_time_right = [round(ic_right[i+1] - ic_right[i], 3) for i in range(len(ic_right) - 1)]

            # For stance times, calculate the time from each IC to the first FC that occurs before the next IC on the same foot.
            fc_left = np.sort(fc[fc["rl_label"] == "left"]["onset"].to_numpy())
            fc_right = np.sort(fc[fc["rl_label"] == "right"]["onset"].to_numpy())

            stance_time_left = []
            for i in range(len(ic_left) - 1):
                t_ic = ic_left[i]              # Current left IC time
                t_next = ic_left[i + 1]        # Next left IC time

                # Select FC events for the left foot that occur between the current and next left IC
                fc_candidates = fc_left[(fc_left > t_ic) & (fc_left < t_next)]

                # Use the first FC candidate to compute stance time (or np.nan if none exists)
                stance_time_left.append(round(fc_candidates[0] - t_ic, 3) if fc_candidates.size > 0 else np.nan)

            stance_time_right = []
            for i in range(len(ic_right) - 1):
                t_ic = ic_right[i]            # Current right IC time
                t_next = ic_right[i + 1]       # Next right IC time

                # Select FC events for the right foot that occur between the current and next right IC
                fc_candidates = fc_right[(fc_right > t_ic) & (fc_right < t_next)]
                stance_time_right.append(round(fc_candidates[0] - t_ic, 3) if fc_candidates.size > 0 else np.nan)

            # Compute swing times as the difference between stride time and stance time for each foot.
            swing_time_left = [round(st - stc, 3) if not np.isnan(stc) else np.nan 
                               for st, stc in zip(stride_time_left, stance_time_left)]
            
            swing_time_right = [round(st - stc, 3) if not np.isnan(stc) else np.nan 
                                for st, stc in zip(stride_time_right, stance_time_right)]

            # Calculate single and double support times for the left foot
            single_support_left = []
            double_support_left = []

            for i in range(len(ic_left) - 1):
                ic_l1 = ic_left[i]          # Start of current left stride (initial contact)
                ic_l2 = ic_left[i + 1]      # End of current left stride (next IC)

                # Get right ICs and FCs that occur within this left stride
                ic_r_within = ic_right[(ic_right > ic_l1) & (ic_right < ic_l2)]
                fc_r_within = fc_right[(fc_right > ic_l1) & (fc_right < ic_l2)]
                fc_l_within = fc_left[(fc_left > ic_l1) & (fc_left < ic_l2)]

                if len(ic_r_within) >= 1 and len(fc_r_within) >= 1 and len(fc_l_within) >= 1:
                    ic_r = ic_r_within[0]   # First right IC after current left IC
                    fc_r = fc_r_within[0]   # First right FC after current left IC
                    fc_l = fc_l_within[0]   # First left FC after current left IC

                    # Double support phase 1: from left IC to right FC (both feet on ground)
                    ds1 = fc_r - ic_l1

                    # Double support phase 2: from right IC to left FC (again, both feet on ground)
                    ds2 = fc_l - ic_r

                    # Total double support time for the left stride
                    double_support = round(ds1 + ds2, 3)
                    double_support_left.append(double_support)

                    # Single support = stance time - double support
                    ss = round(stance_time_left[i] - double_support, 3) if not np.isnan(stance_time_left[i]) else np.nan
                    single_support_left.append(ss)
                else:
                    # If required events are missing, fill with NaN
                    double_support_left.append(np.nan)
                    single_support_left.append(np.nan)

            # Repeat for the right foot
            single_support_right = []
            double_support_right = []

            for i in range(len(ic_right) - 1):
                ic_r1 = ic_right[i]         # Start of current right stride (initial contact)
                ic_r2 = ic_right[i + 1]     # End of current right stride (next IC)

                # Get left ICs and FCs that occur within this right stride
                ic_l_within = ic_left[(ic_left > ic_r1) & (ic_left < ic_r2)]
                fc_l_within = fc_left[(fc_left > ic_r1) & (fc_left < ic_r2)]
                fc_r_within = fc_right[(fc_right > ic_r1) & (fc_right < ic_r2)]

                if len(ic_l_within) >= 1 and len(fc_l_within) >= 1 and len(fc_r_within) >= 1:
                    ic_l = ic_l_within[0]   # First left IC after current right IC
                    fc_l = fc_l_within[0]   # First left FC after current right IC
                    fc_r = fc_r_within[0]   # First right FC after current right IC

                    # Double support phase 1: from right IC to left FC
                    ds1 = fc_l - ic_r1

                    # Double support phase 2: from left IC to right FC
                    ds2 = fc_r - ic_l

                    # Total double support time for the right stride
                    double_support = round(ds1 + ds2, 3)
                    double_support_right.append(double_support)

                    # Single support = stance time - double support
                    ss = round(stance_time_right[i] - double_support, 3) if not np.isnan(stance_time_right[i]) else np.nan
                    single_support_right.append(ss)
                else:
                    # If required events are missing, fill with NaN
                    double_support_right.append(np.nan)
                    single_support_right.append(np.nan)

            # Calculate cadence: total number of IC events per minute over the gait sequence
            all_ics = np.sort(np.concatenate([ic_left, ic_right]))
            duration = all_ics[-1] - all_ics[0] if all_ics.size > 1 else np.nan
            cadence = round(all_ics.size / duration * 60, 2) if duration > 0 else np.nan

            # Append the computed temporal parameters for the current gait sequence into the list
            temporal_parameters_df.append({
                "gait_sequence_id": seq_idx,
                "step_time_l": step_time_l,  # L→R transitions
                "step_time_r": step_time_r,  # R→L transitions
                "stride_time_l": stride_time_left,
                "stride_time_r": stride_time_right,
                "stance_time_l": stance_time_left,
                "stance_time_r": stance_time_right,
                "swing_time_l": swing_time_left,
                "swing_time_r": swing_time_right,
                "single_support_time_l": single_support_left,
                "double_support_time_l": double_support_left,
                "single_support_time_r": single_support_right,
                "double_support_time_r": double_support_right,
                "cadence": cadence
            })

        # Convert the list of dictionaries into a DataFrame and store it in the instance variable
        self.temporal_parameters_ = pd.DataFrame(temporal_parameters_df)

        # Return self
        return self
    

    # Function to calculate gait temporophasic parameters
    def temporophasic_parameters(self) -> pd.DataFrame:
        """
        Calculates temporophasic parameters as percentages of the gait cycle for each foot.

        Returns:
            A DataFrame with temporophasic parameters, including:

                - stance_time_pct_gc_l / r: Stance time as % of the gait cycle.
                - swing_time_pct_gc_l / r: Swing time as % of the gait cycle.
                - single_support_pct_gc_l / r: Single support time as % of the gait cycle.
                - double_support_pct_gc_l / r: Double support time as % of the gait cycle.
        """
        if self.temporal_parameters_ is None:
            raise ValueError("Call temporal_parameters() before computing temporophasic_parameters().")
        
        # Initialize an empty list to store the gait temporophasic parameters for each gait sequence
        temporophasic_parameters_df = []

        for _, row in self.temporal_parameters_.iterrows():
            # Calculate stance and swing times as percentage of stride time
            stance_time_pct_gc_l = [
                (stance / stride) * 100 if stride > 0 else np.nan
                for stance, stride in zip(row["stance_time_l"], row["stride_time_l"])
            ]
            stance_time_pct_gc_r = [
                (stance / stride) * 100 if stride > 0 else np.nan
                for stance, stride in zip(row["stance_time_r"], row["stride_time_r"])
            ]
            swing_time_pct_gc_l = [
                100 - pct if pct is not np.nan else np.nan
                for pct in stance_time_pct_gc_l
            ]
            swing_time_pct_gc_r = [
                100 - pct if pct is not np.nan else np.nan
                for pct in stance_time_pct_gc_r
            ]

            # Calculate single and double support percentages
            single_support_pct_gc_l = [
                (ss / stride) * 100 if stride > 0 and ss is not np.nan else np.nan
                for ss, stride in zip(row["single_support_time_l"], row["stride_time_l"])
            ]
            double_support_pct_gc_l = [
                (ds / stride) * 100 if stride > 0 and ds is not np.nan else np.nan
                for ds, stride in zip(row["double_support_time_l"], row["stride_time_l"])
            ]
            single_support_pct_gc_r = [
                (ss / stride) * 100 if stride > 0 and ss is not np.nan else np.nan
                for ss, stride in zip(row["single_support_time_r"], row["stride_time_r"])
            ]
            double_support_pct_gc_r = [
                (ds / stride) * 100 if stride > 0 and ds is not np.nan else np.nan
                for ds, stride in zip(row["double_support_time_r"], row["stride_time_r"])
            ]

            # Append the computed temporophasic parameters for the current gait sequence into the list
            temporophasic_parameters_df.append({
                "gait_sequence_id": row["gait_sequence_id"],
                "stance_time_pct_gc_l": np.round(stance_time_pct_gc_l, 2).tolist(),
                "stance_time_pct_gc_r": np.round(stance_time_pct_gc_r, 2).tolist(),
                "swing_time_pct_gc_l": np.round(swing_time_pct_gc_l, 2).tolist(),
                "swing_time_pct_gc_r": np.round(swing_time_pct_gc_r, 2).tolist(),
                "single_support_pct_gc_l": np.round(single_support_pct_gc_l, 2).tolist(),
                "double_support_pct_gc_l": np.round(double_support_pct_gc_l, 2).tolist(),
                "single_support_pct_gc_r": np.round(single_support_pct_gc_r, 2).tolist(),
                "double_support_pct_gc_r": np.round(double_support_pct_gc_r, 2).tolist(),
            })

        # Convert the list of dictionaries into a DataFrame and store it in the instance variable
        self.temporophasic_parameters_ = pd.DataFrame(temporophasic_parameters_df)

        # Return self
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
        Calculates spatial gait parameters using vertical acceleration data and pre-detected gait events.
        The method estimates vertical displacement for each step using double integration, then applies the
        inverted pendulum model to compute step length. The step length is calculated using the following model:

                        step_length = 2 * sqrt(2 * wearable_height * delta_z - delta_z^2)

        Then, stride lengths are derived from consecutive steps.

        Args:
            accel_data (pd.DataFrame): Acceleration data as a DataFrame containing at least the vertical axis.
            v_acc_col_name (str): Name of the column representing vertical acceleration.
            sampling_freq_Hz (float): Sampling frequency of the signal in Hertz.
            wearable_height (float, optional): Estimated height of the sensor above the ground in meters.
                                               Default is 1.0 meter.

        Returns:
            pd.DataFrame: A DataFrame containing spatial parameters for each gait sequence, including:

                - 'gait_sequence_id': Index of the gait sequence.
                - 'step_length_l': List of left step lengths (in meters).
                - 'step_length_r': List of right step lengths (in meters).
                - 'stride_length_l': List of left stride lengths (in meters).
                - 'stride_length_r': List of right stride lengths (in meters).
        """
        if self.temporal_parameters_ is None or self.gait_sequences is None:
            # Ensure that the necessary data (gait sequences and temporal parameters) has been computed.
            raise ValueError("Call detect() and temporal_parameters() before computing spatial_parameters().")

        # Initialize an empty list to store the gait spatial parameters for each gait sequence
        spatial_parameters_df = []

        # Convert the vertical acceleration data to a NumPy array and remove the mean (detrend).
        acc_vertical = accel_data[v_acc_col_name].to_numpy().copy()

        # Remove the mean from vertical acceleration signal.
        acc_vertical -= np.mean(acc_vertical)

        # Apply a lowpass Butterworth filter to the vertical acceleration signal.
        acc_vertical = preprocessing.lowpass_filter(acc_vertical, method="butter", order=4, fs=sampling_freq_Hz)

        # Loop through each gait sequence.
        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]  # Start time of the gait sequence.
            end = seq["onset"] + seq["duration"]  # End time of the gait sequence.

            # Select the initial contact (IC) events within the current gait sequence.
            ic_seq = self.initial_contacts[
                (self.initial_contacts["onset"] >= start) & 
                (self.initial_contacts["onset"] <= end)
            ]
            
            # If there are fewer than 2 IC events, skip this sequence.
            if len(ic_seq) < 2:
                continue

            # Build a sorted list of IC events as (time, label) tuples based on onset time.
            ic_sorted = sorted(
                ic_seq[["onset", "rl_label"]].to_records(index=False),
                key=lambda x: x[0]
            )

            # Initialize lists to store computed step lengths.
            step_length_r = []  # Step lengths for R→L transitions (from a right IC to the next left IC).
            step_length_l = []  # Step lengths for L→R transitions (from a left IC to the next right IC).
            step_events = []    # To store detailed information: (t1, t2, direction, step_length).

            # Loop over consecutive pairs of IC events.
            for i in range(len(ic_sorted) - 1):
                t1, lab1 = ic_sorted[i]      # Current event's time and label.
                t2, lab2 = ic_sorted[i+1]    # Next event's time and label.

                # Process only if the two consecutive events are from different feet.
                if lab1 != lab2:
                    # Determine the corresponding indices in the acceleration array based on sampling frequency.
                    idx_start = int(t1 * sampling_freq_Hz)
                    idx_end = int(t2 * sampling_freq_Hz)
                    
                    # Ensure valid indices and that the segment length is positive.
                    if idx_end > idx_start and idx_end <= len(acc_vertical):
                        # Extract the acceleration segment between the two events.
                        accel_seg = acc_vertical[idx_start:idx_end]

                        # Compute velocity by integrating the acceleration segment.
                        vel = cumulative_trapezoid(accel_seg, dx=1/sampling_freq_Hz, initial=0)

                        # Compute displacement by integrating the velocity.
                        disp_m = cumulative_trapezoid(vel, dx=1/sampling_freq_Hz, initial=0)

                        # Calculate vertical displacement (delta_z) as the difference between the max and min displacement (meter).
                        delta_z = np.max(disp_m) - np.min(disp_m)
                        try:
                            # Compute step length using the inverted pendulum model.
                            step_len = 2 * np.sqrt(2 * wearable_height * delta_z - delta_z**2)
                        except Exception:
                            step_len = np.nan

                        # Store detailed step event data.
                        step_events.append((t1, t2, lab1 + "->" + lab2, step_len))

                        # Depending on the transition direction, add the step length to the corresponding list.
                        if lab1 == "right" and lab2 == "left":
                            step_length_r.append(step_len)
                        elif lab1 == "left" and lab2 == "right":
                            step_length_l.append(step_len)

            # Compute stride lengths by pairing consecutive step events that form a full gait cycle.
            # For example:
            #   A right stride might be computed from a R→L step followed by a L→R step.
            #   A left stride might be computed from a L→R step followed by a R→L step.
            stride_length_right = []
            stride_length_left = []
            for i in range(len(step_events) - 1):
                current_dir = step_events[i][2]  # Direction of the current step.
                next_dir = step_events[i+1][2]     # Direction of the next step.
                
                if current_dir == "right->left" and next_dir == "left->right":
                    # Sum the two consecutive step lengths to get the stride length for the right foot.
                    stride_length_right.append(round(step_events[i][3] + step_events[i+1][3], 3))

                elif current_dir == "left->right" and next_dir == "right->left":
                    # Sum the two consecutive step lengths to get the stride length for the left foot.
                    stride_length_left.append(round(step_events[i][3] + step_events[i+1][3], 3))

            # Append the computed spatial parameters for the current gait sequence into the list
            spatial_parameters_df.append({
                "gait_sequence_id": seq_idx,
                "step_length_l": np.round(step_length_l, 3).tolist(),
                "step_length_r": np.round(step_length_r, 3).tolist(),
                "stride_length_l": np.round(stride_length_left, 3).tolist(),
                "stride_length_r": np.round(stride_length_right, 3).tolist(),
            })

        # Convert the list of dictionaries into a DataFrame and store it in the instance variable
        self.spatial_parameters_ = pd.DataFrame(spatial_parameters_df)

        # Return self
        return self


    def spatiotemporal_parameters(self) -> pd.DataFrame:
        """
        Calculates spatio-temporal gait parameters using previously computed spatial and temporal parameters.

        Returns:
            pd.DataFrame with the following columns per gait sequence:
                - gait_speed: Mean gait speed (m/s)
                - stride_speed_l: List of left stride speeds (m/s)
                - stride_speed_r: List of right stride speeds (m/s)
        """
        # Ensure that both spatial and temporal parameters have been computed before proceeding.
        if self.spatial_parameters_ is None or self.temporal_parameters_ is None:
            raise ValueError("Call spatial_parameters() and temporal_parameters() before computing spatiotemporal_parameters().")
        
        # Initialize an empty list to store the gait spatio-temporal parameters for each gait sequence
        spatiotemporal_parameters_df = []

        # Loop through each gait sequence using the spatial_parameters DataFrame.
        for i in range(len(self.spatial_parameters_)):
            # Retrieve the corresponding row from spatial_parameters and temporal_parameters.
            spatial_row = self.spatial_parameters_.iloc[i]
            temporal_row = self.temporal_parameters_.iloc[i]

            # Get the gait sequence identifier.
            gait_sequence_id = spatial_row["gait_sequence_id"]

            # Convert stride lengths and stride times into NumPy arrays for vectorized operations.
            stride_lengths_l = np.array(spatial_row["stride_length_l"])
            stride_lengths_r = np.array(spatial_row["stride_length_r"])
            stride_times_l = np.array(temporal_row["stride_time_l"])
            stride_times_r = np.array(temporal_row["stride_time_r"])

            # Calculate Gait Speed
            # Total distance walked is computed as the sum of all step lengths (from both left and right steps).
            total_distance = np.nansum(spatial_row["step_length_l"]) + np.nansum(spatial_row["step_length_r"])

            # Extract all initial contact (IC) times for the current gait sequence.
            # This is determined by selecting IC events within the start and end times of the sequence.
            all_ics = self.initial_contacts[
                (self.initial_contacts["onset"] >= self.gait_sequences.loc[gait_sequence_id, "onset"]) &
                (self.initial_contacts["onset"] <= self.gait_sequences.loc[gait_sequence_id, "onset"] +
                self.gait_sequences.loc[gait_sequence_id, "duration"])
            ]["onset"].sort_values().to_numpy()

            # Compute ambulation time as the duration between the first and last IC event.
            ambulation_time = all_ics[-1] - all_ics[0] if len(all_ics) > 1 else np.nan

             # Gait speed (in m/s) is the total distance divided by the ambulation time.
            gait_speed = total_distance / ambulation_time if ambulation_time > 0 else np.nan

            # Calculate Stride Speed
            # Stride speed for each side is computed as the stride length divided by the stride time.
            stride_speed_l = np.round((stride_lengths_l / stride_times_l), 3)
            stride_speed_r = np.round((stride_lengths_r / stride_times_r), 3)

            # Append the computed spatio-temporal parameters for the current gait sequence into the list
            spatiotemporal_parameters_df.append({
                "gait_sequence_id": gait_sequence_id,
                "gait_speed": round(gait_speed, 3),
                "stride_speed_l": stride_speed_l.tolist(),
                "stride_speed_r": stride_speed_r.tolist()
            })

        # Convert the list of dictionaries into a DataFrame and store it in the instance variable
        self.spatiotemporal_parameters_ = pd.DataFrame(spatiotemporal_parameters_df)

        # Return self
        return self
