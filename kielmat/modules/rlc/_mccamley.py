# Import libraries
import numpy as np
import pandas as pd
from kielmat.utils import preprocessing
from typing import Optional, Any


class MacCamleyInitialContactClassification:
    """
    The McCamley algorithm [1] determines the laterality of initial contacts (ICs) based on the sign of angular velocity, 
    specifically vertical gyro, anterior-posterior gyro, or a combination of both. The sign of the processed signal at the 
    initial contact position is then used to distinguish between left and right ICs.

    Originally, the algorithm relied on the vertical axis signal alone [1]. However, Ullrich et al. [2] demonstrated that 
    incorporating both vertical and anterior-posterior signals enhances detection accuracy. Additionally, instead of simple 
    mean subtraction, they applied a Butterworth bandpass filter to refine the signal.

    If a `KielMATRecording` object is passed, the rl_label will be added to its events. Otherwise, the result can be accessed 
    via `mccamley_df` attribute.

    Methods:
        detect(gyro_data, sampling_freq_Hz, v_gyr_col_name, ap_gyr_col_name, ic_timestamps, signal_type='vertical', recording=None, tracking_system=None
        ):
            Detects initial contact laterality using the McCamley method. 
            Adds results to the recording if provided and stores the labeled events in `mccamley_df`.

    Example:

        >>> detector = MacCamleyInitialContactClassification()
        >>> detector = detector.detect(
                                gyro_data=gyro_df,
                                sampling_freq_Hz=100,
                                v_gyr_col_name="LowerBack_GYRO_x",
                                ap_gyr_col_name="LowerBack_GYRO_z",
                                ic_timestamps=initial_contacts_df,
                                signal_type="vertical",
                                recording=recording,
                                tracking_system="LowerBack"
                                )
        >>> print(detector.mccamley_df)


    References:
        [1] McCamley, John, et al. "An enhanced estimate of initial contact and final contact instants of time ...
        
        [2] Ullrich, Martin, et al. "Machine learning-based distinction of left and right foot contacts in lower back ...
    """

    def __init__(
        self,
        lowcut: float = 0.5,
        highcut: float = 2,
        order: int = 4,

    ):
        """
        Initializes the McCamley instance with configurable constants.

        Args:
            lowcut (float): The lower cutoff frequency for the bandpass filter (Hz). Default is 0.5 Hz.
            highcut (float): The upper cutoff frequency for the bandpass filter (Hz). Default is 2 Hz.
            order (int): The order of the Butterworth filter. Default is 4.
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

    def detect(
        self,
        gyro_data: pd.DataFrame,
        sampling_freq_Hz: float,
        v_gyr_col_name: str,
        ap_gyr_col_name: str,
        ic_timestamps: pd.DataFrame,
        signal_type: str = "vertical",
        recording: Optional[Any] = None,
        tracking_system: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect initial contact laterality using the McCamley method and optionally update a recording object.

        Args:
            gyro_data (pd.DataFrame): Gyroscope data (N, 3) in SI unit (deg/s).
            sampling_freq_Hz (float): Sampling frequency in Hz.
            v_gyr_col_name (str): Column name for vertical gyroscope signal.
            ap_gyr_col_name (str): Column name for anterior-posterior gyroscope signal.
            ic_timestamps (pd.DataFrame): DataFrame with at least an 'onset' column in seconds.
            signal_type (str): 'vertical', 'anterior_posterior', or 'combined'. Default is 'vertical'.
            recording (Optional[Any]): KielMATRecording object. If provided, will add rl_label column.
            tracking_system (Optional[str]): Required if recording is provided.

        Returns:
            pd.DataFrame: DataFrame with onset, duration, event_type, rl_label, tracking_system
        """
        # Extract vertical and anterior-posterior gyroscope data
        gyro_vertical = gyro_data[v_gyr_col_name].to_numpy()
        gyro_ap = gyro_data[ap_gyr_col_name].to_numpy()

        # Remove mean bias from vertical and anterior-posterior signals
        gyro_vertical = gyro_vertical - np.mean(gyro_vertical)
        gyro_ap = gyro_ap - np.mean(gyro_ap)
        
        # Apply Butterworth bandpass filtering to both signals
        filtered_gyro_vertical = preprocessing.bandpass_filter(gyro_vertical, sampling_rate=sampling_freq_Hz, lowcut=self.lowcut, highcut=self.highcut, order=self.order)
        filtered_gyro_ap = preprocessing.bandpass_filter(gyro_ap, sampling_rate=sampling_freq_Hz, lowcut=self.lowcut, highcut=self.highcut, order=self.order)
        
        # Compute the combined gyroscope signal as (vertical - anterior-posterior)
        gyro_combined = filtered_gyro_vertical - filtered_gyro_ap

        # Signal selection
        signal_map = {
            "vertical": filtered_gyro_vertical,
            "anterior_posterior": filtered_gyro_ap,
            "combined": gyro_combined
        }
        signal = signal_map.get(signal_type)
        if signal is None:
            raise ValueError("Invalid signal_type. Choose 'vertical', 'anterior_posterior', or 'combined'.")

        # Convert onset times to indices
        ic_indices = (ic_timestamps["onset"].to_numpy() * sampling_freq_Hz).astype(int)
        ic_indices = [idx for idx in ic_indices if idx < len(gyro_data)]

        # Assign labels
        if signal_type == "anterior_posterior":
            labels = ["right" if signal[idx] <= 0 else "left" for idx in ic_indices]
        else:
            labels = ["left" if signal[idx] <= 0 else "right" for idx in ic_indices]

        # Create output
        self.ic_rl_list_ = pd.DataFrame({
            "onset": ic_timestamps["onset"].values,
            "rl_label": labels
        })

        # If recording is provided, add rl_label to it
        if recording is not None:
            if tracking_system is None:
                raise ValueError("If 'recording' is provided, 'tracking_system' must also be specified.")

            df = recording.events[tracking_system]

            # Map labels to the appropriate rows
            label_map = dict(zip(self.ic_rl_list_["onset"], self.ic_rl_list_["rl_label"]))
            mask = df["event_type"] == "initial contact"
            recording.events[tracking_system].loc[mask, "rl_label"] = df.loc[mask, "onset"].map(label_map)

            # Reorder columns
            cols = recording.events[tracking_system].columns.tolist()
            if "rl_label" in cols and "tracking_system" in cols:
                cols.remove("rl_label")
                idx = cols.index("event_type") + 1
                cols.insert(idx, "rl_label")
                recording.events[tracking_system] = recording.events[tracking_system][cols]

            # Return just the updated initial contacts
            self.mccamley_df = recording.events[tracking_system][mask]
            return self

        else:
            # If no recording is given, create a full result table from ic_timestamps
            result_df = ic_timestamps.copy()
            result_df["duration"] = 0.0
            result_df["event_type"] = "initial contact"
            result_df["tracking_system"] = tracking_system if tracking_system else "unknown"
            result_df["rl_label"] = labels

            # Reorder columns for consistency
            self.mccamley_df = result_df[["onset", "duration", "event_type", "rl_label", "tracking_system"]]

            return self