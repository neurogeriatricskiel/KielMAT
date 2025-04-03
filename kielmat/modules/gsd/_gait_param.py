import numpy as np
import pandas as pd
from kielmat.utils import preprocessing
from scipy.integrate import cumulative_trapezoid


class GaitSpatioTemporalParameters:
    """
    This algorithm calculates spatio-temporal gait parameters based on pre-detected gait events,
    such as initial contacts (IC), final contacts (FC), and gait sequences. It is designed for
    analyzing human gait using event-based temporal data derived from IMU systems.

    The algorithm uses clinically validated definitions and reference literature to compute
    step time, stride time, swing and stance phases, cadence, and percentage-based temporophasic
    parameters (stance and swing time as a percentage of the gait cycle). The events must be
    pre-identified using other algorithms or manual annotations and passed as pandas DataFrames.

    This implementation uses algorithms based on literature-reported definitions:
    
    - Temporal parameters are calculated following clinical definitions and validated studies [1-4].
    - Temporophasic percentages (stance and swing times) are derived as portions of the gait cycle [2,3].
    - Spatial parameters are derived based on the inverted pendulum model using [5].
        
    Workflow:

    1. Load gait sequences, initial contacts, and final contacts using the `detect()` method.
    2. Call `temporal_parameters()` to compute duration-based gait parameters (e.g., stride time).
    3. Call `temporophasic_parameters()` to derive percent-based stance and swing times.
    4. Call `spatial_parameters()` to estimate step and stride lengths using the inverted pendulum model.

    This implementation supports left and right side separation, handles missing or incomplete
    strides, and skips over incomplete data ranges.

    Methods:
        detect(gait_sequences, initial_contacts, final_contacts):
            Loads event data for later computation.

        temporal_parameters():
            Computes step time, stride time, swing time, stance time, and cadence.

        temporophasic_parameters():
            Computes percentage-based stance and swing times for each gait cycle.

        spatial_parameters():
            Calculates step and stride lengths using vertical acceleration data.
      

    Examples:
        >>> gait_stp = GaitSpatioTemporalParameters()
        >>> gait_stp.detect(gait_sequences, initial_contacts, final_contacts)
        
        >>> temporal_df = gait_stp.temporal_parameters()
        >>> print(temporal_df)
               gait_sequence_id     step_time_l     step_time_r  ...  stance_time_r     cadence
            0  0                    [0.49, 0.5]     [0.51, 0,5]  ...  [0.61,0.6]        98.0

        >>> phasic_df = gait_stp.temporophasic_parameters()
        >>> print(phasic_df)
               gait_sequence_id  stance_time_pct_gc_l  ...  swing_time_pct_gc_r
            0  0                 [62.1, 63.3]          ...  [36.9, 36.3]

        >>> spatial_df = gait_stp.spatial_parameters(
        ...     accel_data=accel_df, 
        ...     v_acc_col_name="pelvis_ACCEL_z", 
        ...     sampling_freq_Hz=100, 
        ...     wearable_height=1.0
        ... )
        >>> print(spatial_df)
               gait_sequence_id     step_length_l       ...     stride_length_r
            0  0                    [0.57, 0.63]        ...     [1.27]
    
    References:
        [1] Zijlstra, W., & At L. Hof (2003). Assessment of spatio-temporal gait parameters from trunk accelerations during human walking.
        
        [2] Moe-Nilssen, R., et al. (2020). Spatiotemporal gait parameters for older adults. Gait & Posture.
        
        [3] Hollman, J. H., et al. (2011). Normative spatiotemporal gait parameters in older adults. Gait & Posture.
        
        [4] Hass, C. J., et al. (2012). Quantitative normative gait data in a large cohort of ambulatory persons with Parkinson’s disease. PLoS ONE.

        [5] Cerny, M., Noury, N., & Deplorte, L. (2015). Validation of Inverted Pendulum model for gait length calculation.

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
          - step_time_l / step_time_r: Time from an initial contact (IC) of one foot to the next IC of the opposite foot.
          - stride_time_l / stride_time_r: Time between consecutive ICs of the same foot.
          - stance_time_l / stance_time_r: Time from an IC to the first final contact (FC) in the same stride.
          - swing_time_l / swing_time_r: Stride time minus stance time.
          - cadence: Total number of steps per minute.
        """
        temporal_results = []

        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]
            end = seq["onset"] + seq["duration"]

            # Select IC and FC events within the sequence
            ic = self.initial_contacts[(self.initial_contacts["onset"] >= start) & 
                                       (self.initial_contacts["onset"] <= end)]
            fc = self.final_contacts[(self.final_contacts["onset"] >= start) & 
                                     (self.final_contacts["onset"] <= end)]

            if len(ic) < 2 or len(fc) < 1:
                continue # Not enough events to analyze

            # Remove misaligned FC if it occurs before the first IC of the same side
            first_ic_time = ic.iloc[0]["onset"]
            first_ic_label = ic.iloc[0]["rl_label"]
            first_fc_time = fc.iloc[0]["onset"]
            first_fc_label = fc.iloc[0]["rl_label"]
            if first_fc_time < first_ic_time and first_fc_label == first_ic_label:
                fc = fc.iloc[1:]

            # Split events into left/right
            ic_left = np.sort(ic[ic["rl_label"] == "left"]["onset"].to_numpy())
            ic_right = np.sort(ic[ic["rl_label"] == "right"]["onset"].to_numpy())
            fc_left = np.sort(fc[fc["rl_label"] == "left"]["onset"].to_numpy())
            fc_right = np.sort(fc[fc["rl_label"] == "right"]["onset"].to_numpy())

            # Step time: from one IC to the next IC of the opposite side
            step_time_left = []
            for t in ic_left:
                next_right = ic_right[ic_right > t]
                if next_right.size > 0:
                    step_time_left.append(next_right[0] - t)

            step_time_right = []
            for t in ic_right:
                next_left = ic_left[ic_left > t]
                if next_left.size > 0:
                    step_time_right.append(next_left[0] - t)

            # Stride time: from one IC to the next IC on the same side
            stride_time_left = np.diff(ic_left) if ic_left.size > 1 else np.array([])
            stride_time_right = np.diff(ic_right) if ic_right.size > 1 else np.array([])

            # Stance time: IC to first FC before the next IC
            stance_time_left = []
            for i in range(len(ic_left) - 1):
                t_ic = ic_left[i]
                t_next = ic_left[i + 1]
                fc_candidates = fc_left[(fc_left > t_ic) & (fc_left < t_next)]
                stance_time_left.append(fc_candidates[0] - t_ic if fc_candidates.size > 0 else np.nan)

            stance_time_right = []
            for i in range(len(ic_right) - 1):
                t_ic = ic_right[i]
                t_next = ic_right[i + 1]
                fc_candidates = fc_right[(fc_right > t_ic) & (fc_right < t_next)]
                stance_time_right.append(fc_candidates[0] - t_ic if fc_candidates.size > 0 else np.nan)

            # Swing time = Stride - Stance
            swing_time_left = []
            for i in range(len(stride_time_left)):
                if i < len(stance_time_left) and not np.isnan(stance_time_left[i]):
                    swing_time_left.append(stride_time_left[i] - stance_time_left[i])
                else:
                    swing_time_left.append(np.nan)

            swing_time_right = []
            for i in range(len(stride_time_right)):
                if i < len(stance_time_right) and not np.isnan(stance_time_right[i]):
                    swing_time_right.append(stride_time_right[i] - stance_time_right[i])
                else:
                    swing_time_right.append(np.nan)

            # Cadence = total steps per minute
            all_ics = np.sort(np.concatenate([ic_left, ic_right]))
            duration = all_ics[-1] - all_ics[0] if all_ics.size > 1 else np.nan
            cadence = (all_ics.size / duration * 60) if duration > 0 else np.nan

            # Collect result
            temporal_results.append({
                "gait_sequence_id": seq_idx,
                "step_time_l": np.round(np.array(step_time_left), 3).tolist(),
                "step_time_r": np.round(np.array(step_time_right), 3).tolist(),
                "stride_time_l": np.round(stride_time_left, 3).tolist(),
                "stride_time_r": np.round(stride_time_right, 3).tolist(),
                "stance_time_l": np.round(np.array(stance_time_left), 3).tolist(),
                "stance_time_r": np.round(np.array(stance_time_right), 3).tolist(),
                "swing_time_l": np.round(np.array(swing_time_left), 3).tolist(),
                "swing_time_r": np.round(np.array(swing_time_right), 3).tolist(),
                "cadence": round(cadence, 2)
            })

        # Store results in a DataFrame
        self.temporal_parameters_ = pd.DataFrame(temporal_results)

        # Return self
        return self

    # Function to calculate gait temporophasic parameters
    def temporophasic_parameters(self) -> pd.DataFrame:
        """
        Calculates temporophasic parameters as percentages of the gait cycle for each foot.

        Returns:
            A DataFrame with temporophasic parameters, including:

                - stance_time_pct_gc_l: Stance time as % of the gait cycle for the left foot.
                - stance_time_pct_gc_r: Stance time as % of the gait cycle for the right foot.
                - swing_time_pct_gc_l: Swing time as % of the gait cycle for the left foot.
                - swing_time_pct_gc_r: Swing time as % of the gait cycle for the right foot.
        """
        if self.temporal_parameters_ is None:
            raise ValueError("Call temporal_parameters() before computing temporophasic_parameters().")

        # Temporophasic parameters
        temporophasic_parameters_list = []

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

            # Collect result
            temporophasic_parameters_list.append({
                "gait_sequence_id": row["gait_sequence_id"],
                "stance_time_pct_gc_l": np.round(stance_time_pct_gc_l, 2).tolist(),
                "stance_time_pct_gc_r": np.round(stance_time_pct_gc_r, 2).tolist(),
                "swing_time_pct_gc_l": np.round(swing_time_pct_gc_l, 2).tolist(),
                "swing_time_pct_gc_r": np.round(swing_time_pct_gc_r, 2).tolist(),
            })

        # Store results in a DataFrame
        self.temporophasic_parameters_ = pd.DataFrame(temporophasic_parameters_list)

        # Return self
        return self

    # Function to calculate spatial gait parameters
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
            raise ValueError("Call detect() and temporal_parameters() before computing spatial_parameters().")
        
        spatial_results = []
        
        # Preprocess acceleration: remove offset and filter
        acc_vertical = accel_data[v_acc_col_name].to_numpy().copy()
        acc_vertical -= np.mean(acc_vertical)
        acc_vertical = preprocessing.lowpass_filter(acc_vertical, method="butter", order=4, fs=sampling_freq_Hz)
        
        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]
            end = seq["onset"] + seq["duration"]
            
            ic_seq = self.initial_contacts[(self.initial_contacts["onset"] >= start) &
                                        (self.initial_contacts["onset"] <= end)]
            fc_seq = self.final_contacts[(self.final_contacts["onset"] >= start) &
                                        (self.final_contacts["onset"] <= end)]
            if len(ic_seq) < 2 or len(fc_seq) < 1:
                continue
            
            # Align events if needed.
            first_ic_time = ic_seq.iloc[0]["onset"]
            first_ic_label = ic_seq.iloc[0]["rl_label"]
            first_fc_time = fc_seq.iloc[0]["onset"]
            first_fc_label = fc_seq.iloc[0]["rl_label"]
            if first_fc_time < first_ic_time and first_fc_label == first_ic_label:
                fc_seq = fc_seq.iloc[1:]
            
            # Separate left/right initial and final contacts.
            ic_left = ic_seq[ic_seq["rl_label"] == "left"]["onset"].to_numpy()
            ic_right = ic_seq[ic_seq["rl_label"] == "right"]["onset"].to_numpy()
            fc_left = fc_seq[fc_seq["rl_label"] == "left"]["onset"].to_numpy()
            fc_right = fc_seq[fc_seq["rl_label"] == "right"]["onset"].to_numpy()
            
            # Compute step lengths with timestamps for left foot.
            left_steps = []   # Each entry is (timestamp, step_length)
            for t in ic_left:
                # Find the first final contact (FC) that occurs after the current initial contact (IC)
                future_fc = fc_left[fc_left > t]
                if future_fc.size == 0:
                    continue
                
                # Use the first FC that follows this IC
                t_fc = future_fc[0]

                # Convert event times to indices in the acceleration array
                start_idx = int(t * sampling_freq_Hz)
                end_idx = int(t_fc * sampling_freq_Hz)
                if end_idx > start_idx:
                    # Extract the vertical acceleration segment from IC to FC
                    accel_segment = acc_vertical[start_idx:end_idx]

                    # First integration: compute vertical velocity using cumulative trapezoidal integration
                    velocity = cumulative_trapezoid(accel_segment, dx=1/sampling_freq_Hz, initial=0)

                    # Second integration: compute vertical displacement from velocity
                    displacement = cumulative_trapezoid(velocity, dx=1/sampling_freq_Hz, initial=0)

                    # Calculate peak-to-peak vertical displacement during the step
                    delta_z = np.max(displacement) - np.min(displacement)

                    # Estimate step length using inverted pendulum model
                    # step_length = 2 * sqrt(2 * h * delta_z - delta_z^2)
                    disc = 2 * wearable_height * delta_z - delta_z**2
                    step_length = 2 * np.sqrt(disc) if disc > 0 else 0

                    # Store result
                    left_steps.append((t, step_length))
            
            # Compute step lengths with timestamps for right foot.
            right_steps = []  # Each entry is (timestamp, step_length)
            for t in ic_right:
                # Find the first final contact (FC) that occurs after the current initial contact (IC)
                future_fc = fc_right[fc_right > t]
                if future_fc.size == 0:
                    continue

                # Use the first FC that follows this IC
                t_fc = future_fc[0]

                # Convert event times to indices in the acceleration array
                start_idx = int(t * sampling_freq_Hz)
                end_idx = int(t_fc * sampling_freq_Hz)
                if end_idx > start_idx:
                    # Extract the vertical acceleration segment from IC to FC
                    accel_segment = acc_vertical[start_idx:end_idx]

                    # First integration: compute vertical velocity using cumulative trapezoidal integration
                    velocity = cumulative_trapezoid(accel_segment, dx=1/sampling_freq_Hz, initial=0)

                    # Second integration: compute vertical displacement from velocity
                    displacement = cumulative_trapezoid(velocity, dx=1/sampling_freq_Hz, initial=0)

                    # Calculate peak-to-peak vertical displacement during the step
                    delta_z = np.max(displacement) - np.min(displacement)

                    # Estimate step length using inverted pendulum model
                    # step_length = 2 * sqrt(2 * h * delta_z - delta_z^2)
                    disc = 2 * wearable_height * delta_z - delta_z**2
                    step_length = 2 * np.sqrt(disc) if disc > 0 else 0

                    # Store result
                    right_steps.append((t, step_length))
            
            # Form stride lengths using timestamp alignment.
            # Left stride: from left IC to next left IC, with a right step in between.
            stride_length_left = []
            for i in range(len(left_steps) - 1):
                # Current left initial contact and its step length
                t_left, left_step = left_steps[i]

                # Next left initial contact
                t_next_left, _ = left_steps[i+1]

                # Find the right step with timestamp between the two left contacts.
                candidates = [r for r in right_steps if r[0] > t_left and r[0] < t_next_left]
                if candidates:
                    # Use the first candidate (assuming one per cycle).
                    stride_length_left.append(left_step + candidates[0][1]) # Sum left + right step lengths
            
            # Right stride: from right IC to next right IC, with a left step in between.
            stride_length_right = []
            for i in range(len(right_steps) - 1):
                # Current right initial contact and its step length
                t_right, right_step = right_steps[i]

                # Next right initial contact
                t_next_right, _ = right_steps[i+1]
                candidates = [l for l in left_steps if l[0] > t_right and l[0] < t_next_right]
                if candidates:
                    # Use the first candidate (assuming one per cycle).
                    stride_length_right.append(right_step + candidates[0][1]) # Sum right + left step lengths
            
            spatial_results.append({
                "gait_sequence_id": seq_idx,
                "step_length_l": np.round([s[1] for s in left_steps], 3).tolist(),
                "step_length_r": np.round([s[1] for s in right_steps], 3).tolist(),
                "stride_length_l": np.round(np.array(stride_length_left), 3).tolist(),
                "stride_length_r": np.round(np.array(stride_length_right), 3).tolist(),
            })
        
        # Store results as a DataFrame
        self.spatial_parameters_ = pd.DataFrame(spatial_results)

        # Return self 
        return self

