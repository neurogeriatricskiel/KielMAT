"""Inertial Orientation Estimation."""
import numpy as np
from vqf import VQF
from ..base import BaseAlgorithm


class IOE(BaseAlgorithm):
    def __init__(self):
        pass

    def apply(self, data: np.ndarray, sampling_freq_Hz: float) -> np.ndarray:
        # Parse input args
        gyr = np.ascontiguousarray(data[:, 3:6])
        acc = np.ascontiguousarray(data[:, 0:3])
        Ts = 1. / sampling_freq_Hz

        # Initialize the VQF algorithm
        vqf = VQF(Ts)
        out = vqf.updateBatch(gyr, acc)
        gyr_rot = np.zeros_like(gyr)
        acc_rot = np.zeros_like(acc)
        for i in range(len(out["quat6D"])):
            gyr_rot[i, :] = vqf.quatRotate(q=out["quat6D"][i], v=gyr[i, :])
            acc_rot[i, :] = vqf.quatRotate(q=out["quat6D"][i], v=acc[i, :])
        rotated_data = np.c_[acc_rot, gyr_rot]
        return rotated_data