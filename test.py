import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from ngmt.datasets import keepcontrol, mobilised
from ngmt.modules.gsd import GSDB
from ngmt.modules.icd import ICDA
import h5py

def main():
    # User settings
    FILE_NAME = 'C:\\Users\\Project\\Desktop\\Gait_Sequence\\Mobilise-D dataset_1-18-2023\\PFF\\data.mat'

    # Load IMU data from the MATLAB file
    ds = mobilised.load_file(file_name=FILE_NAME)

    # Load accelerometer data (imu_acc) and sampling frequency (fs)
    imu_acc = [rec for rec in ds.devices[0].recordings if rec.type=='Acc'][0].data
    sampling_frequency = [rec for rec in ds.devices[0].recordings if rec.type=='Acc'][0].fs

    # Use Gait_Sequence_Detection to detect gait sequence 
    gsd_output = GSDB.Gait_Sequence_Detection(imu_acceleration=imu_acc, sampling_frequency=sampling_frequency, plot_results=True)
    
    # Use Initial_Contact_Detection to detect intial contacts
    icd_output = ICDA.Initial_Contact_Detection(imu_acceleration=imu_acc, gait_sequences=gsd_output, sampling_frequency=sampling_frequency, plot_results=True)
    
    return


if __name__ == "__main__":
    main()