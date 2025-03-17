# Import libraries
import numpy as np
import pandas as pd
from kielmat.utils import preprocessing


class MacCamleyInitialContactClassification:
    """
    The McCamley algorithm [1] determines the laterality of initial contacts (ICs) based on the sign of angular velocity, 
    specifically vertical gyro, anterior-posterior gyro, or a combination of both. The sign of the processed signal at the 
    initial contact position is then used to distinguish between left and right ICs.

    Originally, the algorithm relied on the vertical axis signal alone [1]. However, Ullrich et al. [2] demonstrated that 
    incorporating both vertical and anterior-posterior signals enhances detection accuracy. Additionally, instead of simple 
    mean subtraction, they applied a Butterworth bandpass filter to refine the signal.

    Finally, initial contact information is provided as a DataFrame with columns `onset` and `rl_label`.

    Methods:
        detect(gyro_data, sampling_freq_Hz, v_gyr_col_name, ap_gyr_col_name, ic_timestamps, signal_type):
            Detects initial contact laterality using the McCamley method.

    Examples:
        Detect initial contacts and classify laterality:

        >>> detector = McCamleyLateralityDetection()
        >>> detector = detector.detect(
        >>>     gyro_data=gyro_dataframe,
        >>>     sampling_freq_Hz=100,
        >>>     v_gyr_col_name="LowerBack_GYRO_x",
        >>>     ap_gyr_col_name="LowerBack_GYRO_z",
        >>>     ic_timestamps=initial_contact_df,
        >>>     signal_type="vertical"
        >>> )
        >>> print(detector.ic_rl_list_)
                onset    rl_label
            0   5.000    left
            1   5.600    right

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
    ) -> pd.DataFrame:
        """
        Detect initial contact laterality using the McCamley method.
        
        Args:
            gyro_data (pd.DataFrame): Gyroscope data (N, 3) in SI unit (deg/s).
            sampling_freq_Hz (float): Sampling frequency in Hz.
            v_gyr_col_name (str): Column name for vertical gyroscope signal.
            ap_gyr_col_name (str): Column name for anterior-posterior gyroscope signal.
            ic_timestamps (pd.DataFrame): A DataFrame containing detected initial contact (IC) events. 
                This should include an `onset` column with initial contact timestamps in seconds. 
                The ICs can be detected using any suitable gait event detection algorithm.
            signal_type (str, optional): Signal type for classification. Options:
                - 'vertical': Uses only the vertical gyroscope signal.
                - 'anterior_posterior': Uses only the anterior-posterior gyroscope signal.
                - 'combined': Uses the difference between vertical and anterior-posterior signals (gyro_vertical - gyro_ap).
                Default is 'vertical'.
        
        Returns:
            pd.DataFrame: DataFrame with initial contact onset and their corresponding labels as right or left.
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

        # Convert initial contact timestamps to indices
        ic_indices = (ic_timestamps["onset"].to_numpy() * sampling_freq_Hz).astype(int)
        ic_indices = [idx for idx in ic_indices if idx < len(gyro_data)]

        # Select the appropriate signal type for classification
        signal_map = {
            "vertical": filtered_gyro_vertical,
            "anterior_posterior": filtered_gyro_ap,
            "combined": gyro_combined
        }
        signal = signal_map.get(signal_type)
        if signal is None:
            raise ValueError("Invalid signal type. Choose 'vertical', 'anterior_posterior', or 'combined'.")
                
        # Classify initial contacts using McCamley method (classify based on sign of the signal)
        labels = ["left" if signal[idx] <= 0 else "right" for idx in ic_indices]
        
        # Store results in DataFrame
        self.ic_rl_list_ = pd.DataFrame({"onset": ic_timestamps["onset"].values, "rl_label": labels})

        return self