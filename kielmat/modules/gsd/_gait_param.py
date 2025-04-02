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

    ### Workflow:
    1. Load gait sequences, initial contacts, and final contacts using the `detect()` method.
    2. Call `temporal_parameters()` to compute duration-based gait parameters (e.g., stride time).
    3. Call `temporophasic_parameters()` to derive percent-based stance and swing times.

    This implementation supports left and right side separation, handles missing or incomplete
    strides, and skips over incomplete data ranges.

    Methods:
        detect(gait_sequences, initial_contacts, final_contacts):
            Loads event data for later computation.

        temporal_parameters():
            Computes step time, stride time, swing time, stance time, and cadence.

        temporophasic_parameters():
            Computes percentage-based stance and swing times for each gait cycle.


    Examples:
        >>> gait_stp = GaitSpatioTemporalParameters()
        >>> gait_stp.detect(gait_sequences, initial_contacts, final_contacts)
        >>> temporal_df = gait_stp.temporal_parameters()
        >>> phasic_df = gait_stp.temporophasic_parameters()
        >>> print(temporal_df)
               gait_sequence_id  step_time_l  step_time_r  ...  stance_time_r  cadence
            0               0      [0.49]       [0.51]     ...      [0.61]       98.0

        >>> print(phasic_df)
               gait_sequence_id  stance_time_pct_gc_l  ...  swing_time_pct_gc_r

    References:
        [1] Zijlstra, W., & At L. Hof (2004). Assessment of spatio-temporal gait parameters from trunk accelerations during human walking
        
        [2] Moe-Nilssen, R., et al. (2020). Spatiotemporal gait parameters for older adults. Gait & Posture.
        
        [3] Hollman, J. H., et al. (2011). Normative spatiotemporal gait parameters in older adults. Gait & Posture.
        
        [4] Hass, C. J., et al. (2012). Quantitative normative gait data in a large cohort of ambulatory persons with Parkinson’s disease. PLoS ONE.
    """

    def __init__(
        self,
     
    ):
        """
        Initializes the GaitSpatioTemporalParameters instance.
        """
        self.temporal_parameters = None
        self.temporophasic_parameters = None


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

    # Function to calculate temporal parameters
    def temporal_parameters(self) -> pd.DataFrame:
        """
        Calculates temporal gait parameters using the provided initial and final contact events.

        Returns:
            pd.DataFrame: Gait temporal parameters per sequence, including:
                - step_time_l / step_time_r: time between initial contact of one foot and opposite foot
                - stride_time_l / stride_time_r: time between two successive ICs of the same foot
                - swing_time_l / swing_time_r: time from final contact to next initial contact of same foot
                - stance_time_l / stance_time_r: time from initial contact to final contact of same foot
                - cadence: total steps per minute
        """
        if self.gait_sequences is None or self.initial_contacts is None or self.final_contacts is None:
            raise ValueError("Gait sequences and contact events must be loaded using the detect() method.")

        # Temporal parameters
        temporal_parameters = []

        for seq_idx, seq in self.gait_sequences.iterrows():
            start = seq["onset"]
            end = seq["onset"] + seq["duration"]

            # Filter events within the current gait sequence
            ic = self.initial_contacts[
                (self.initial_contacts["onset"] >= start) &
                (self.initial_contacts["onset"] <= end)
            ]
            fc = self.final_contacts[
                (self.final_contacts["onset"] >= start) &
                (self.final_contacts["onset"] <= end)
            ]

            # Separate IC and FC by side
            ic_l = ic[ic["rl_label"] == "left"]["onset"].to_numpy()
            ic_r = ic[ic["rl_label"] == "right"]["onset"].to_numpy()
            fc_l = fc[fc["rl_label"] == "left"]["onset"].to_numpy()
            fc_r = fc[fc["rl_label"] == "right"]["onset"].to_numpy()

            # Skip if there's less than 2 total steps
            if len(ic_l) + len(ic_r) < 2:
                continue

            # Step Times: time between the initial contact of one foot and the initial contact of the opposite foot
            step_time_l = [ic_r[i] - ic_l[i] for i in range(min(len(ic_l), len(ic_r)))]
            step_time_r = [ic_l[i + 1] - ic_r[i] for i in range(len(ic_r) - 1)]

            # Stride Times: time between two successive initial contacts of the same foot
            stride_time_l = [ic_l[i + 1] - ic_l[i] for i in range(len(ic_l) - 1)]
            stride_time_r = [ic_r[i + 1] - ic_r[i] for i in range(len(ic_r) - 1)]

            # Swing Times: time from the initial contact to next the initial contact on the same foot
            swing_time_l = [
                ic_l[i] - fc_l[i]
                for i in range(min(len(ic_l), len(fc_l)))
            ]

            swing_time_r = [
                ic_r[i] - fc_r[i]
                for i in range(min(len(ic_r), len(fc_r)))
            ]

            # Stance Times = calculated as stride time - swing time
            stance_time_l = [
                stride_time_l[i] - swing_time_l[i]
                for i in range(min(len(stride_time_l), len(swing_time_l)))
            ]
            stance_time_r = [
                stride_time_r[i] - swing_time_r[i]
                for i in range(min(len(stride_time_r), len(swing_time_r)))
            ]

            # Cadence (steps/min): number of steps per minute.
            all_ics = np.sort(np.concatenate([ic_l, ic_r]))
            duration = all_ics[-1] - all_ics[0] if len(all_ics) > 1 else None
            cadence = (len(all_ics) / duration) * 60 if duration and duration > 0 else np.nan

            # Append parameters
            temporal_parameters.append({
                "gait_sequence_id": seq_idx,
                "step_time_l": np.round(step_time_l, 3).tolist(),
                "step_time_r": np.round(step_time_r, 3).tolist(),
                "stride_time_l": np.round(stride_time_l, 3).tolist(),
                "stride_time_r": np.round(stride_time_r, 3).tolist(),
                "swing_time_l": np.round(swing_time_l, 3).tolist(),
                "swing_time_r": np.round(swing_time_r, 3).tolist(),
                "stance_time_l": np.round(stance_time_l, 3).tolist(),
                "stance_time_r": np.round(stance_time_r, 3).tolist(),
                "cadence": round(cadence, 2),
            })

        # Store results in a DataFrame
        self.temporal_parameters_ = pd.DataFrame(temporal_parameters)

        return self.temporal_parameters_

    # Function to calculate temporophasic parameters
    def temporophasic_parameters(self):
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

            temporophasic_parameters_list.append({
                "gait_sequence_id": row["gait_sequence_id"],
                "stance_time_pct_gc_l": np.round(stance_time_pct_gc_l, 2).tolist(),
                "stance_time_pct_gc_r": np.round(stance_time_pct_gc_r, 2).tolist(),
                "swing_time_pct_gc_l": np.round(swing_time_pct_gc_l, 2).tolist(),
                "swing_time_pct_gc_r": np.round(swing_time_pct_gc_r, 2).tolist(),
            })

        # Store results in a DataFrame
        self.temporophasic_parameters_ = pd.DataFrame(temporophasic_parameters_list)

        return self.temporophasic_parameters_