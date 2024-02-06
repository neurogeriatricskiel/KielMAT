import pytest
import numpy as np
import pandas as pd
from ngmt.modules.gsd._paraschiv import ParaschivIonescuGaitSequenceDetection



# Test data
acceleration_data = pd.read_csv('.\example_lower_back_acc.csv')
sampling_freq_Hz = 100

def test_detect():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()
    # Call the detect method
    gsd.detect(data=acceleration_data, sampling_freq_Hz=sampling_freq_Hz)
    gait_sequences = gsd.gait_sequences_

    # Assertions
    assert isinstance(gait_sequences, pd.DataFrame), "Gait sequences should be a DataFrame."
    assert 'onset' in gait_sequences.columns, "Gait sequences should have 'onset' column."
    assert 'duration' in gait_sequences.columns, "Gait sequences should have 'duration' column."
    assert 'event_type' in gait_sequences.columns, "Gait sequences should have 'event_type' column."
    assert 'tracking_systems' in gait_sequences.columns, "Gait sequences should have 'tracking_systems' column."
    assert 'tracked_points' in gait_sequences.columns, "Gait sequences should have 'tracked_points' column."

def test_invalid_input_data():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid input data
    invalid_data = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    with pytest.raises(ValueError):
        gsd.detect(data=invalid_data, sampling_freq_Hz=sampling_freq_Hz)

def test_invalid_sampling_freq():
    # Initialize the class
    gsd = ParaschivIonescuGaitSequenceDetection()

    # Test with invalid sampling frequency
    invalid_sampling_freq = 'invalid'
    with pytest.raises(ValueError):
        gsd.detect(data=acceleration_data, sampling_freq_Hz=invalid_sampling_freq)

if __name__ == "__main__":
    test_detect()
    test_invalid_input_data()
    test_invalid_sampling_freq()
    print("All tests passed!")