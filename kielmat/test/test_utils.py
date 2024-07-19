import os
import h5py
import numpy as np
import pytest
from scipy import io as sio
from tempfile import NamedTemporaryFile
from kielmat.utils.matlab_loader import HDF5Decoder,  convert_mat_to_dict, load_matlab

@pytest.fixture
def temp_files():
    temp_mat_file = NamedTemporaryFile(delete=False, suffix='.mat')
    temp_h5_file = NamedTemporaryFile(delete=False, suffix='.h5')
    yield temp_mat_file.name, temp_h5_file.name
    try:
        os.unlink(temp_mat_file.name)
    except PermissionError:
        pass  # Handle file still in use error on Windows
    try:
        os.unlink(temp_h5_file.name)
    except PermissionError:
        pass  # Handle file still in use error on Windows

def test_initialization():
    decoder = HDF5Decoder(load_only_su=True)
    assert decoder.load_only_su

    decoder = HDF5Decoder(load_only_su=False)
    assert not decoder.load_only_su

def test_mat2dict_h5(temp_files):
    _, temp_h5_file = temp_files
    with h5py.File(temp_h5_file, 'w') as f:
        f.create_dataset('a', data=[1, 2, 3])
        f.create_dataset('b', data=[4, 5, 6])

    decoder = HDF5Decoder()
    with h5py.File(temp_h5_file, 'r') as f:
        with pytest.raises(NotImplementedError):
            decoder.mat2dict(f)

def test_convert_mat(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, 'w') as f:
        dataset = f.create_dataset('test_dataset', data=np.array([1, 2, 3], dtype='float64'))
        dataset.attrs['MATLAB_class'] = np.string_('double')

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, 'r') as f:
        result = decoder.convert_mat(f['test_dataset'], flatten_keys=False)

    np.testing.assert_array_equal(result, [1, 2, 3])

if __name__ == '__main__':
    pytest.main()