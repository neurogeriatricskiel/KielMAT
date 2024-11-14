import numpy as np
import pywt
from ..base import BaseAlgorithm


class CwtFilter(BaseAlgorithm):
    def __init__(self, wavelet_name: str = "gaus1"):
        self.wavelet_name = wavelet_name
    
    def apply(self, data: np.ndarray, sampling_freq_Hz: float, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Call the continuous wavelet transform from PyWavelets
        coefs, frequencies = pywt.cwt(
            data=data,
            scales=scales,
            wavelet=self.wavelet_name,
            sampling_period=1. / sampling_freq_Hz
        )
        return coefs, frequencies