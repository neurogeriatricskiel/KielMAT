import numpy as np
import scipy.signal
from typing import Sequence
from ..base import BaseAlgorithm


class ButterworthFilter(BaseAlgorithm):

    def __init__(self, order: int, cutoff_freq_Hz: float | Sequence[float], btype: str = "lowpass"):
        self.order = order
        self.cutoff_freq_Hz = cutoff_freq_Hz
        self.btype = btype

    def apply(self, data: np.ndarray, sampling_freq_Hz: float) -> np.ndarray:
        # Get the filter coefficients
        b, a = scipy.signal.butter(self.order, self.cutoff_freq_Hz, btype=self.btype, fs=sampling_freq_Hz)

        # Apply the filter
        filtered_data = scipy.signal.filtfilt(b, a, data, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1))
        return filtered_data