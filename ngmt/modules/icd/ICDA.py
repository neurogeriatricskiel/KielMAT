# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage
import pywt
from ngmt.utils import preprocessing

def Initial_Contact_Detection(imu_acceleration, gait_sequences, sampling_frequency, plot_results):
    """_summary_
    Initial Contact Detection Algorithm (ICDA) performs Signal Decomposition on low back IMU accelerometer data for detecting initial contacts (ICs).

    Parameters:
        imu_acceleration (numpy.ndarray): IMU accelerometer data.
        gait_sequences (list): List of dictionaries containing gait sequence 'Start' and 'End' fields.
        sampling_frequency (float): Sampling frequency of the accelerometer data.
        plot_results (bool): Whether to plot the results.

    Returns:
        list: List of dictionaries containing detected ICs and associated information.
    """
    # Extract vertical accelerometer data
    acc_vertical = imu_acceleration[:, 0]

    # Process each gait sequence
    if isinstance(gait_sequences, list) and all(isinstance(seq, dict) and 'Start' in seq and 'End' in seq for seq in gait_sequences):
        processed_output = []

        max_ic_count = max(len(gs.get('IC', [])) for gs in gait_sequences)

        for j in range(len(gait_sequences)):
            start_index = round(sampling_frequency * gait_sequences[j]['Start'])
            stop_index = round(sampling_frequency * gait_sequences[j]['End'])
            accV_gs = acc_vertical[start_index:stop_index]

            try:
                # Perform Signal Decomposition Algorithm for Initial Contacts (ICs)
                IC_rel, _ = preprocessing.signal_decomposition_algorithm(accV_gs, sampling_frequency)
                IC = gait_sequences[j]['Start'] + IC_rel
                gait_sequences[j]['IC'] = IC.tolist()
            except Exception as e:
                print('SD algorithm did not run successfully. Returning an empty vector of ICs')
                print(f'Error: {e}')
                IC = []
                gait_sequences[j]['IC'] = []

            # Add sampling frequency and pad ICs to the output
            gait_sequences[j]['SamplingFrequency'] = sampling_frequency  # Add sampling frequency to the output
            gait_sequences[j]['IC'] += [np.nan] * (max_ic_count - len(gait_sequences[j]['IC']))

        processed_output = gait_sequences
    else:
        processed_output = []

    if plot_results:
        # Load FIR filter coefficients
        filtering_file = scipy.io.loadmat('C:\\Users\\Project\\Desktop\\Gait_Sequence\\Mobilise-D-TVS-Recommended-Algorithms\\GSDB\\Library\\FIR-2-3Hz-40.mat')
        filter_coeffs = filtering_file['Num'][0, :]

        target_sampling_frequency = 40
        max_ic_count = max(len(gs.get('IC', [])) for gs in gait_sequences)

        IC_all_signal = np.vstack([
            gs.get('IC', []) + [np.nan] * (max_ic_count - len(gs.get('IC', [])))
            for gs in gait_sequences if 'IC' in gs
        ])

        IC_indices = np.round(IC_all_signal * sampling_frequency).astype(int)

        # Resample and filter accelerometer data
        accV_resampled = preprocessing.resample_interpolate(acc_vertical, sampling_frequency, target_sampling_frequency)
        accV_filtered = scipy.signal.filtfilt(filter_coeffs, 1, preprocessing.remove_40Hz_drift(accV_resampled))
        accV_integral = scipy.integrate.cumtrapz(accV_filtered) / target_sampling_frequency

        # Perform Continuous Wavelet Transform (CWT)
        num_scales = 9
        wavelet = 'gaus2'
        sampling_period = 1 / target_sampling_frequency
        coefficients, _ = pywt.cwt(accV_integral, np.arange(1, num_scales + 1), wavelet, sampling_period)
        desired_scale = 9
        accV_cwt = coefficients[desired_scale - 1, :]
        accV_cwt = accV_cwt - np.mean(accV_cwt)
        
        # Resample CWT results back to original sampling frequency
        accV_processed = preprocessing.resample_interpolate(accV_cwt, target_sampling_frequency, sampling_frequency)

        valid_IC_indices = IC_indices[(IC_indices >= 0) & (IC_indices < len(accV_processed))]
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(accV_processed)), accV_processed, 'b', linewidth=3)
        plt.plot(valid_IC_indices, accV_processed[valid_IC_indices], 'ro', markersize=8)  # Plot ICs as red points
        plt.legend(['Processed Vertical Acceleration Signal', 'Initial Contacts (IC)'], fontsize=16)
        plt.xlabel('Sample Index', fontsize=16)
        plt.ylabel('Amplitude', fontsize=16)
        plt.title('Processed Acceleration Signal and Detected Initial Contacts', fontsize=16)
        plt.grid(True)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
        plt.show()

    return processed_output