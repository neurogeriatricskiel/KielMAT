# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import scipy.ndimage
import pywt
from ngmt.utils import preprocessing


def gsd_low_back_acc(imu_acc, sampling_frequency):
    """_summary_
    Perform Gait Sequence Detection (GSD) using low back accelerometer data.

    Args:
        imu_acc (numpy.ndarray): Input accelerometer data (N, 3) for x, y, and z axes.
        sampling_frequency (float): Sampling frequency of the accelerometer data.

    Returns:
        list: A list of dictionaries containing gait sequence information, including start and end times, and sampling frequency.
    """
    gsd_output = {}

    # Calculate the norm of acceleration as acc_n using x, y, and z components.
    acc_n = np.sqrt(imu_acc[:, 0]**2 + imu_acc[:, 1]**2 + imu_acc[:, 2]**2)    
    
    # Resample acc_n to target sampling frequency using resample_interpolate function.
    fs_initial = sampling_frequency    # Initial sampling frequency of the acceleration data
    algorithm_target_fs = 40      # Targeted sampling frequency of the acceleration data
    acc_n_40 = preprocessing.resample_interpolate(acc_n, fs_initial, algorithm_target_fs)      # Resampled data with 40Hz  
    
    # Applying Savitzky-Golay filter to smoothen the resampled data with frequency of 40Hz
    window_length = 21         
    polynomial_order = 7
    acc_n_filt1 = scipy.signal.savgol_filter(acc_n_40, window_length, polynomial_order)

    # Load FIR filter designed and apply for the low SNR, impaired, asymmetric, and slow gait
    filtering_file =  scipy.io.loadmat('C:\\Users\\Project\\Desktop\\Gait_Sequence\\Mobilise-D-TVS-Recommended-Algorithms\\GSDB\\Library\\FIR-2-3Hz-40.mat')
    num = filtering_file['Num'][0, :]
    
    # Remove drifts using defined function in utls (RemoveDrift40Hz).
    # Define parameters of the filter
    numerator_coefficient = num
    denominator_coefficient = np.array([1., ])
    acc_n_filt2 = scipy.signal.filtfilt(numerator_coefficient, denominator_coefficient, preprocessing.remove_40Hz_drift(acc_n_filt1))
    
    # Perform the continuous wavelet transform on the filtered acceleration data accN_filt2
    scales = 10                      #  At scale=10 the wavelet is stretched by a factor of 10, making it sensitive to lower frequencies in the signal.
    wavelet = 'gaus2'                #  The Gaussian wavelets ("gausP" where P is an integer between 1 and and 8) correspond to the Pth order derivatives of the function
    sampling_period = 1/algorithm_target_fs     #  Sampling period which is equal to 1/algorithm_target_fs
    coefficients, _ = pywt.cwt(acc_n_filt2, np.arange(1, scales + 1), wavelet, sampling_period)
    desired_scale = 10               # Choose the desired scale you want to access (1 to scales) and extract it from the coefficients
    acc_n_filt3 = coefficients[desired_scale - 1, :]
    
    # Applying Savitzky-Golay filter to further smoothen the wavelet transformed data
    window_length = 11
    polynomial_order = 5
    acc_n_filt4 = scipy.signal.savgol_filter(acc_n_filt3, window_length, polynomial_order)
    
    # Perform continuous wavelet transform
    coefficients, _ = pywt.cwt(acc_n_filt4, np.arange(1, scales + 1), wavelet, sampling_period)
    desired_scale = 10  # Choose the desired scale you want to access (1 to scales) and extract it from the coefficients
    acc_n_filt5 = coefficients[desired_scale - 1, :]

    # Smoothing the data using successive Gaussian filters from scipy.ndimage
    sigma_1 = 1.9038      # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt5,'gaussian',window);
    sigma_2 = 1.9038      # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt6,'gaussian',window);
    sigma_3 = 2.8936      # The sigma_1 = 2.8936 gives the same results when window=15 in the MATLAB fuction smoothdata(accN_filt7,'gaussian',window);
    sigma_4 = 1.9038      # The sigma_1 = 1.9038 gives the same results when window=10 in the MATLAB fuction smoothdata(accN_filt8,'gaussian',window);
    sigma_values = [sigma_1, sigma_2, sigma_3, sigma_4]            # Vectors of sigma values for successive Gaussian filters
    acc_n_filt6  = scipy.ndimage.gaussian_filter(acc_n_filt5, sigma=sigma_values[0], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=None)
    acc_n_filt7  = scipy.ndimage.gaussian_filter(acc_n_filt6 , sigma=sigma_values[1], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=None)
    acc_n_filt8  = scipy.ndimage.gaussian_filter(acc_n_filt7, sigma=sigma_values[2], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=None)
    acc_n_filt9  = scipy.ndimage.gaussian_filter(acc_n_filt8 , sigma=sigma_values[3], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, radius=None)
    
    # Use processed acceleration data for further analysis.
    signal_detected_activity  = acc_n_filt9 

    # Compute the envelope of the processed acceleration data.
    alarm = []
    alarm, _ = preprocessing.calculate_envelope_activity(signal_detected_activity, int(round(algorithm_target_fs)), 1, int(round(algorithm_target_fs)), 0) 

    # Initialize a list for walking bouts.
    walk_low_back = []

    # Process alarm data to identify walking bouts.
    if alarm.size > 0:
        temp = np.where(alarm > 0)[0]  # Find nonzeros
        idx = preprocessing.find_consecutive_groups(alarm > 0)
        for j in range(len(idx)):
            if idx[j, 1] - idx[j, 0] <= 3 * algorithm_target_fs:
                alarm[idx[j, 0]:idx[j, 1] + 1] = 0
            else:
                walk_low_back.extend(signal_detected_activity[idx[j, 0]:idx[j, 1] + 1])
                
        # Convert walk_low_back list to a NumPy array
        walk_low_back_array = np.array(walk_low_back)
                
        # Find positive peaks in the walk_low_back_array
        pksp, _ = scipy.signal.find_peaks(walk_low_back_array, height=0)
                
        # Get the corresponding y-axis data values for the positive peak
        pksp = walk_low_back_array[pksp]
                
        # Find negative peaks in the inverted walk_low_back array
        pksn , _ = scipy.signal.find_peaks(-walk_low_back_array)
                
        # Get the corresponding y-axis data values for the positive peak
        pksn = -walk_low_back_array[pksn]
                
        # Convert pksn list to a NumPy array before using it in concatenation
        pksn_array = np.array(pksn)
                
        # Combine positive and negative peaks
        pks = np.concatenate((pksp, pksn_array))
                
        # Calculate the data adaptive threshold using the 5th percentile of the combined peaks
        threshold = np.percentile(pks, 5)
    
        # Set f to sigDetActv
        f = signal_detected_activity

    else:
        threshold = 0.15  # If hilbert envelope fails to detect 'active', try version [1]
        f = acc_n_filt4

    # Detect mid-swing peaks.
    min_peaks, max_peaks = preprocessing.find_local_min_max(f, threshold)

    # Find pulse trains in max_peaks and remove ones with steps less than 4
    t1 = preprocessing.identify_pulse_trains(max_peaks)
    
    # Access the fields of the struct-like array
    t1 = [train for train in t1 if train['steps'] >= 4]
    
    # Find pulse trains in min_peaks and remove ones with steps less than 4
    t2 = preprocessing.identify_pulse_trains(min_peaks)
    
    # Access the fields of the struct-like array
    t2 = [train for train in t2 if train['steps'] >= 4]
    
    # Convert t1 and t2 to sets and find their intersection
    t_final = preprocessing.find_interval_intersection(preprocessing.convert_pulse_train_to_array(t1), preprocessing.convert_pulse_train_to_array(t2))

    return gsd_output