import os
import sys
import scipy
from ..utils.data_utils import (IMUDataset, IMUDevice, IMURecording)
from ..utils import matlab_loader as matlab_loader

# Dictionary that maps sensor types to their corresponding units of measurement
_MAP_UNITS = {'Acc': 'g',         # Accelerometer data: units of gravity ('g')
              'Gyr': 'deg/s',     # Gyroscope data: degrees per second ('deg/s')
              'Mag': 'microTesla',# Magnetometer data: microteslas ('microTesla')
              'Bar': 'hPa'}       # Barometer data: hectopascals ('hPa')

def load_file(file_name: str) -> IMUDataset:
    
    # Load data from the MATLAB file
    data_dict = matlab_loader.load_matlab(file_name, top_level="data")

    # Create an IMUDataset object to store data
    imu_dataset = IMUDataset(subject_id='')

    # Get a list of IMU devices
    tracked_points = list(data_dict['TimeMeasure1']['Recording4']['SU'].keys())

    # Loop over the tracked points
    for tracked_point in tracked_points:

        # Instantiate an IMUDevice object for the tracked point
        imu_device = IMUDevice(tracked_point=tracked_point) 

        # Loop over the recordings (e.g., acc, gyr, mag, etc)
        for sensor_type in data_dict['TimeMeasure1']['Recording4']['SU'][tracked_point].keys():
            if sensor_type not in  ['Timestamp', 'Fs']:

                # Get the data
                data = data_dict['TimeMeasure1']['Recording4']['SU'][tracked_point][sensor_type]

                # Get the corresponding sampling frequency
                fs = data_dict['TimeMeasure1']['Recording4']['SU'][tracked_point]['Fs'][sensor_type]

                # Append recording to IMUDevice object
                imu_device.recordings.append(
                    IMURecording(
                        type=sensor_type,
                        units=_MAP_UNITS[sensor_type],
                        fs=fs,
                        data=data
                    )
                )

        # Add the IMUDevice to the IMUDataset
        imu_dataset.devices.append(imu_device)

    return imu_dataset