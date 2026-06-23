# Import libraries
from typing import Optional
import numpy as np
import pandas as pd
import scipy.signal


class LeeuwenFinalContactDetection:
    """
    The Leeuwen final contact detection algorithm identifies final contacts (toe-offs)
    from optical motion-capture marker data using a coordinate-based approach.

    The algorithm works on the anterior-posterior (AP) position of a heel marker expressed relative
    to a reference marker on the pelvis (e.g. sacrum). Subtracting the pelvis position removes whole-body
    translation, so the resulting signal reflects heel motion relative to the centre of mass. The
    relative signal is zero-centred and low-pass filtered with a zero-phase Butterworth filter. Final
    contacts coincide with the most posterior position of the heel, i.e. the negative peaks of the
    relative signal, which are located with prominence- and distance-constrained peak detection on the
    inverted signal.

    The algorithm processes one leg per call: the user passes the heel marker column and the pelvis
    reference column for the side of interest, and the detected side is recorded in the output. To
    obtain events for both legs, call :meth:`detect` once per side and combine the results.

    Finally, the final contacts information is provided as a DataFrame with columns `onset`,
    `duration`, `event_type`, `side`, and `tracking_system`.

    Methods:
        detect(data, sampling_freq_Hz, heel_col_name, reference_col_name, ...):
            Detects final contacts from the marker position data.

    Examples:
        Find final contacts of the left leg from heel and pelvis markers

        >>> fcd = LeeuwenFinalContactDetection()
        >>> fcd = fcd.detect(
        ...     data=marker_data,
        ...     sampling_freq_Hz=100,
        ...     heel_col_name="LHEE_PosY",
        ...     reference_col_name="SACR_PosY",
        ...     side="left",
        ... )
        >>> print(fcd.final_contacts_)
                onset   duration   event_type      side   tracking_system
            0   1.78    0          final contact   left   omc
            1   2.89    0          final contact   left   omc

    References:
        [1] Zeni et al. (2008). Two simple methods for determining gait events during treadmill and
            overground walking. Gait & Posture, 27(4), 710-714. https://doi.org/10.1016/j.gaitpost.2007.07.007
    """

    def __init__(
        self,
    ):
        """
        Initializes the LeeuwenFinalContactDetection instance.
        """
        self.final_contacts_ = None

    def detect(
        self,
        data: pd.DataFrame,
        sampling_freq_Hz: float,
        heel_col_name: str,
        reference_col_name: str,
        side: Optional[str] = None,
        cutoff_freq_Hz: float = 6.0,
        min_step_time_s: float = 0.5,
        prominence_factor: float = 1.0,
        dt_data: Optional[pd.Series] = None,
        tracking_system: Optional[str] = None,
    ) -> "LeeuwenFinalContactDetection":
        """
        Detects final contacts (toe-offs) based on the input marker position data.

        Args:
            data (pd.DataFrame): Input marker position data containing at least the heel and reference columns.
            sampling_freq_Hz (float): Sampling frequency of the marker data in Hz.
            heel_col_name (str): Column name of the heel marker, anterior-posterior axis (e.g. "LHEE_PosY").
            reference_col_name (str): Column name of the pelvis/sacrum reference marker, anterior-posterior axis (e.g. "SACR_PosY").
            side (str, optional): The leg the heel marker belongs to (e.g. "left" or "right"). Recorded in the events. Default is None.
            cutoff_freq_Hz (float, optional): Low-pass Butterworth cutoff frequency in Hz. Default is 6.0.
            min_step_time_s (float, optional): Minimum time between consecutive final contacts in seconds, used as the minimum peak distance. Default is 0.5.
            prominence_factor (float, optional): Peak prominence as a multiple of the filtered signal's standard deviation. Default is 1.0.
            dt_data (pd.Series, optional): Original datetime in the input data. If provided, the output onset will be based on that.
            tracking_system (str, optional): Tracking system the data is from, to be used for the events DataFrame. Default is None.

        Returns:
            LeeuwenFinalContactDetection: Returns an instance of the class.
                The final contacts information is stored in the 'final_contacts_' attribute,
                which is a pandas DataFrame in BIDS-like format with the following columns:
                    - onset: Time of the final contact (seconds, or datetime if dt_data is provided).
                    - duration: Duration of the event (always 0 for instantaneous contacts).
                    - event_type: Type of the event (always 'final contact').
                    - side: The leg the event belongs to.
                    - tracking_system: Tracking system the events are derived from.
        """
        # Error handling for invalid input data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        if not isinstance(sampling_freq_Hz, (int, float)) or sampling_freq_Hz <= 0:
            raise ValueError("Sampling frequency must be a positive float.")

        if not isinstance(heel_col_name, str) or not isinstance(
            reference_col_name, str
        ):
            raise ValueError("heel_col_name and reference_col_name must be strings.")

        if side is not None and not isinstance(side, str):
            raise ValueError("side must be a string.")

        if tracking_system is not None and not isinstance(tracking_system, str):
            raise ValueError("tracking_system must be a string")

        # check if dt_data is a pandas Series with datetime values
        if dt_data is not None and (
            not isinstance(dt_data, pd.Series)
            or not pd.api.types.is_datetime64_any_dtype(dt_data)
        ):
            raise ValueError("dt_data must be a pandas Series with datetime values")

        # check if dt_data is provided and if it is a series with the same length as data
        if dt_data is not None and len(dt_data) != len(data):
            raise ValueError("dt_data must be a series with the same length as data")

        # Return an empty result if the data is empty
        if data.empty:
            self.final_contacts_ = pd.DataFrame()
            return self

        # Check that the requested columns are present
        for col in (heel_col_name, reference_col_name):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the input data.")

        # Compute the pelvis-referenced heel signal and remove the DC offset, so that the
        # signal reflects heel motion relative to the centre of mass (see References).
        relative_signal = (
            data[heel_col_name].to_numpy() - data[reference_col_name].to_numpy()
        )
        relative_signal = relative_signal - np.mean(relative_signal)

        # Apply a 4th-order zero-phase Butterworth low-pass filter (matches the source method).
        b, a = scipy.signal.butter(
            4, cutoff_freq_Hz / (sampling_freq_Hz / 2), btype="low"
        )
        filtered_signal = scipy.signal.filtfilt(b, a, relative_signal)

        # Final contacts are the most posterior heel positions -> negative peaks of the signal,
        # i.e. positive peaks of the inverted signal.
        min_distance = max(1, int(min_step_time_s * sampling_freq_Hz))
        prominence = np.std(filtered_signal) * prominence_factor
        fc_samples, _ = scipy.signal.find_peaks(
            -filtered_signal,
            distance=min_distance,
            prominence=prominence if prominence > 0 else None,
        )

        # Build the BIDS-like events DataFrame
        onsets = fc_samples / sampling_freq_Hz
        self.final_contacts_ = pd.DataFrame(
            {
                "onset": onsets,
                "duration": 0,
                "event_type": "final contact",
                "side": side,
                "tracking_system": tracking_system,
            }
        )

        # If original datetime is available, use it for the 'onset' column
        if dt_data is not None:
            self.final_contacts_["onset"] = dt_data.iloc[fc_samples].reset_index(
                drop=True
            )

        return self
