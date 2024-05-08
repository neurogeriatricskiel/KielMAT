import h5py
import pandas as pd
from ngmt.utils.ngmt_dataclass import NGMTRecording

def load_recording(file_name: str, tracking_systems: list[str], tracked_points: dict[str, list[str]]) -> pd.DataFrame:
    """
    Load a recording from the SensePark file.

    Args:
        file_name (str): The path to the HDF5 file containing the recording data.
        tracking_systems (list of str): A list of tracking systems for which data are to be returned.
        tracked_points (dict): A dictionary defining for each tracking system the tracked points of interest.

    Returns:
        NGMTRecording: An instance of NGMTRecording containing the loaded data and channel information.
    """
    # Initialize data and channels dictionaries
    data_dict = {}
    channels_dict = {}

    # Open the HDF5 file
    with h5py.File(file_name, 'r') as h5_file:
        # Loop through each tracking system
        for tracking_sys in tracking_systems:
            # Get the case ID corresponding to the monitor label
            monitor_label = tracked_points[tracking_sys][0]
            case_id = None
            for ix, label in enumerate(h5_file.attrs['MonitorLabelList']):
                if label.decode("utf-8") == monitor_label:
                    case_id = h5_file.attrs['CaseIdList'][ix].decode("utf-8")
                    break
            
            if case_id is None:
                print(f"Error: Case ID not found for monitor label '{monitor_label}'")
                continue

            # Get the sample rate
            sample_rate = h5_file[case_id].attrs['SampleRate']
            
            # Raw data
            raw_acc = h5_file[case_id]['Calibrated']['Accelerometers'][:]
            raw_gyro = h5_file[case_id]['Calibrated']['Gyroscopes'][:]
            raw_magn = h5_file[case_id]['Calibrated']['Magnetometers'][:]
            
            # Construct channel names
            channel_names = [
                f'{monitor_label}_Acc_x', f'{monitor_label}_Acc_y', f'{monitor_label}_Acc_z',
                f'{monitor_label}_Gyro_x', f'{monitor_label}_Gyro_y', f'{monitor_label}_Gyro_z',
                f'{monitor_label}_Magn_x', f'{monitor_label}_Magn_y', f'{monitor_label}_Magn_z'
            ]
            
            # Append channel data to the channels dictionary
            channel_data = {
                "name": channel_names,
                "component": ["x", "y", "z"] * 3,
                "type": ["ACCEL", "GYRO", "MAGN"] * 3,
                "tracked_point": [monitor_label] * len(channel_names),
                "units": ["m/s^2", 'rad/s', "µT"] * 3,
                "sampling_frequency": [sample_rate] * len(channel_names)
            }
            channels_dict[tracking_sys] = pd.DataFrame(channel_data)

            # Add data to the data dictionary
            data = {
                f'{monitor_label}_Acc_x': raw_acc[:, 0],
                f'{monitor_label}_Acc_y': raw_acc[:, 1],
                f'{monitor_label}_Acc_z': raw_acc[:, 2],
                f'{monitor_label}_Gyro_x': raw_gyro[:, 0],
                f'{monitor_label}_Gyro_y': raw_gyro[:, 1],
                f'{monitor_label}_Gyro_z': raw_gyro[:, 2],
                f'{monitor_label}_Magn_x': raw_magn[:, 0],
                f'{monitor_label}_Magn_y': raw_magn[:, 1],
                f'{monitor_label}_Magn_z': raw_magn[:, 2]
            }
            data_dict[tracking_sys] = pd.DataFrame(data)

    # Return NGMTRecording instance
    return NGMTRecording(data=data_dict, channels=channels_dict)
