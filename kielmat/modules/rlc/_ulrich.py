import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from kielmat.utils import preprocessing
import joblib


class UllrichInitialContactClassification:
    """
    The Ullrich algorithm [1] classifies the laterality of initial contacts (ICs) as either 'left' or 'right' 
    based on gyroscope signals collected from a lower-back inertial measurement unit (IMU). Unlike traditional 
    rule-based approaches such as the McCamley method [2], this algorithm relies on a supervised machine 
    learning model trained on labeled walking data.This classification approach is particularly useful in studies 
    involving gait analysis, where precise identification of foot contacts is essential for deriving 
    spatio-temporal gait parameters.

    At the core of the method is a feature extraction process that computes a six-dimensional feature vector 
    for each IC timestamp. This includes the vertical and anterior-posterior components of the gyroscope signal, 
    their first derivatives (approximating angular velocity changes), and second derivatives (representing 
    rotational acceleration). These signals are preprocessed using a Butterworth bandpass filter to remove 
    noise and low-frequency drift.

    The extracted features are passed to a pre-trained machine learning model, which can be a random forest 
    (rfc), support vector machine (svm_linear or svm_rbf), or k-nearest neighbors (knn), depending on user 
    preference. The model assigns a label of either 'left' or 'right' to each initial contact. The trained 
    models must be stored in the `ml_models/` folder within the module directory.

    If a `KielMATRecording` object is provided, the method adds the predicted laterality labels to the 
    corresponding initial contact events in the recording’s event structure. Otherwise, the labeled events 
    are stored in the `ulrich_df` attribute as a standalone DataFrame.

    Methods:
        detect(gyro_data, sampling_freq_Hz, v_gyr_col_name, ap_gyr_col_name, ic_timestamps, ml_model_type, recording=None, tracking_system=None):
            Detects initial contact laterality using the Ullrich ML method and stores the labeled events.

    Example:

        >>> classifier = UllrichInitialContactClassification()
        >>> classifier = classifier.detect(
                gyro_data=gyro_df,
                sampling_freq_Hz=100,
                v_gyr_col_name="LowerBack_GYRO_x",
                ap_gyr_col_name="LowerBack_GYRO_z",
                ic_timestamps=initial_contacts_df,
                ml_model_type="rfc",
                recording=recording,
                tracking_system="SU"
        )
        >>> print(classifier.ulrich_df)

    References:
        [1] McCamley, John, et al. "An enhanced estimate of initial contact and final contact instants of time ...
        
        [2] Ullrich, Martin, et al. "Machine learning-based distinction of left and right foot contacts in lower back ...
    """

    def __init__(self, lowcut: float = 0.5, highcut: float = 2, order: int = 4):
        """
        Initializes the Ullrich classifier with signal filtering settings.

        Args:
            lowcut (float): Lower cutoff frequency for bandpass filter (Hz). Default: 0.5 Hz.
            highcut (float): Upper cutoff frequency for bandpass filter (Hz). Default: 2 Hz.
            order (int): Order of the Butterworth filter. Default: 4.
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
        ml_model_type: str,
        recording: Optional[Any] = None,
        tracking_system: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Detect initial contact laterality using a pre-trained ML model from Ullrich et al. (2021).

        Args:
            gyro_data (pd.DataFrame): Gyroscope data with columns for vertical and AP axes.
            sampling_freq_Hz (float): Sampling frequency in Hz.
            v_gyr_col_name (str): Column name of vertical gyroscope signal.
            ap_gyr_col_name (str): Column name of anterior-posterior gyroscope signal.
            ic_timestamps (pd.DataFrame): DataFrame with an 'onset' column in seconds.
            ml_model_type (str): One of ['rfc', 'svm_linear', 'svm_rbf', 'knn'].
            recording (Optional[Any]): KielMATRecording object to update with rl_label (optional).
            tracking_system (Optional[str]): Required if recording is provided.

        Returns:
            pd.DataFrame: DataFrame with onset, duration, event_type, rl_label, and tracking_system.
        """
        # Extract signals
        gyro_v = gyro_data[v_gyr_col_name].to_numpy()
        gyro_ap = gyro_data[ap_gyr_col_name].to_numpy()

        # Remove DC bias
        gyro_v -= np.mean(gyro_v)
        gyro_ap -= np.mean(gyro_ap)

        # Apply Butterworth bandpass filter
        filtered_v = preprocessing.bandpass_filter(gyro_v, sampling_freq_Hz, self.lowcut, self.highcut, self.order)
        filtered_ap = preprocessing.bandpass_filter(gyro_ap, sampling_freq_Hz, self.lowcut, self.highcut, self.order)

        # Convert onset timestamps to indices
        ic_indices = (ic_timestamps["onset"].to_numpy() * sampling_freq_Hz).astype(int)
        ic_indices = [idx for idx in ic_indices if idx < len(filtered_v)]

        # Extract features: [v, ap, v', ap', v'', ap'']
        features = []
        for idx in ic_indices:
            features.append([
                filtered_v[idx],
                filtered_ap[idx],
                np.gradient(filtered_v)[idx],
                np.gradient(filtered_ap)[idx],
                np.gradient(np.gradient(filtered_v))[idx],
                np.gradient(np.gradient(filtered_ap))[idx],
            ])
        features = np.array(features)

        # Load the pre-trained model
        try:
            model_path = Path(__file__).resolve().parent / "ml_models" / f"{ml_model_type}_model.pkl"
        except NameError:
            model_path = Path.cwd() / "kielmat" / "modules" / "rlc" / "ml_models" / f"{ml_model_type}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"ML model not found: {model_path}")
        model = joblib.load(model_path)

        # Predict left/right label
        predictions = model.predict(features)
        labels = [str(p).strip().lower() for p in predictions]

        # Prepare output DataFrame
        self.ic_rl_list_ = pd.DataFrame({
            "onset": ic_timestamps["onset"].values,
            "rl_label": labels
        })

        # Attach results to KielMATRecording if provided
        if recording is not None:
            if tracking_system is None:
                raise ValueError("If 'recording' is provided, 'tracking_system' must also be specified.")
            
            df = recording.events[tracking_system]
            label_map = dict(zip(self.ic_rl_list_["onset"], self.ic_rl_list_["rl_label"]))
            mask = df["event_type"] == "initial contact"
            df.loc[mask, "rl_label"] = df.loc[mask, "onset"].map(label_map)

            # Reorder columns for consistency
            cols = df.columns.tolist()
            if "rl_label" in cols and "tracking_system" in cols:
                cols.remove("rl_label")
                idx = cols.index("event_type") + 1
                cols.insert(idx, "rl_label")
                recording.events[tracking_system] = df[cols]

            self.ulrich_df = df[mask]
            return self

        else:
            result_df = ic_timestamps.copy()
            result_df["duration"] = 0.0
            result_df["event_type"] = "initial contact"
            result_df["tracking_system"] = tracking_system if tracking_system else "unknown"
            result_df["rl_label"] = labels
            self.ulrich_df = result_df[["onset", "duration", "event_type", "rl_label", "tracking_system"]]
            return self