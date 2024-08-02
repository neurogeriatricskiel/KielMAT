import os
import h5py
import numpy as np
import pytest
from scipy import io as sio
from tempfile import NamedTemporaryFile
from kielmat.utils.matlab_loader import HDF5Decoder, convert_mat_to_dict, load_matlab
import importlib.resources as resources


@pytest.fixture
def temp_files():
    temp_mat_file = NamedTemporaryFile(delete=False, suffix=".mat")
    temp_h5_file = NamedTemporaryFile(delete=False, suffix=".h5")

    # Ensure files are closed before tests start
    temp_mat_file.close()
    temp_h5_file.close()

    yield temp_mat_file.name, temp_h5_file.name

    # Clean up the temporary files
    os.remove(temp_mat_file.name)
    os.remove(temp_h5_file.name)


def test_initialization():
    decoder = HDF5Decoder(load_only_su=True)
    assert decoder.load_only_su

    decoder = HDF5Decoder(load_only_su=False)
    assert not decoder.load_only_su


def test_mat2dict_mat(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        f.create_dataset("a", data=[1, 2, 3])
        f.create_dataset("b", data=[4, 5, 6])
        f.attrs["MATLAB_class"] = np.string_("struct")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.mat2dict(f)
        assert "a" in result
        assert "b" in result


def test_mat2dict_h5(temp_files):
    _, temp_h5_file = temp_files
    with h5py.File(temp_h5_file, "w") as f:
        f.create_dataset("a", data=[1, 2, 3])
        f.create_dataset("b", data=[4, 5, 6])

    decoder = HDF5Decoder()
    with h5py.File(temp_h5_file, "r") as f:
        with pytest.raises(NotImplementedError):
            decoder.mat2dict(f)


def test_unpack_mat(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        group = f.create_group("group")
        dataset = group.create_dataset(
            "dataset", data=np.array([1, 2, 3], dtype="float64")
        )
        dataset.attrs["MATLAB_class"] = np.string_("double")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.unpack_mat(f["group/dataset"], depth=0)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_convert_mat(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        dataset = f.create_dataset(
            "test_dataset", data=np.array([1, 2, 3], dtype="float64")
        )
        dataset.attrs["MATLAB_class"] = np.string_("double")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.convert_mat(f["test_dataset"], flatten_keys=False)

    np.testing.assert_array_equal(result, [1, 2, 3])


def test_has_refs_without_references():
    file_name = "test.h5"
    with h5py.File(file_name, "w") as f:
        dset = f.create_dataset("dataset", data=np.array([[1, 2]]))
        decoder = HDF5Decoder()
        assert decoder._has_refs(dset) is False


def test_convert_mat_char(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        dataset = f.create_dataset(
            "char_dataset", data=np.array([72, 101, 108, 108, 111], dtype="uint8")
        )
        dataset.attrs["MATLAB_class"] = np.string_("char")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.convert_mat(f["char_dataset"], flatten_keys=False)


def test_convert_mat_empty():
    file_name = "test.h5"
    with h5py.File(file_name, "w") as f:
        dset = f.create_dataset("empty", data=np.array([]))
        dset.attrs["MATLAB_class"] = "canonical empty"
        decoder = HDF5Decoder()
        result = decoder.convert_mat(dset, flatten_keys=False)
        assert result is None


def test_convert_mat_logical(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        dataset = f.create_dataset(
            "logical_dataset", data=np.array([1, 0, 1], dtype="uint8")
        )
        dataset.attrs["MATLAB_class"] = np.string_("logical")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.convert_mat(f["logical_dataset"], flatten_keys=False)

    assert np.array_equal(result, [True, False, True])


def test_convert_mat_empty(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        dataset = f.create_dataset("empty_dataset", data=np.array([], dtype="uint8"))
        dataset.attrs["MATLAB_class"] = np.string_("canonical empty")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.convert_mat(f["empty_dataset"], flatten_keys=False)

    assert result is None


def test_convert_mat_complex(temp_files):
    temp_mat_file, _ = temp_files
    with h5py.File(temp_mat_file, "w") as f:
        complex_data = np.array([(1.0, 2.0)], dtype=[("real", "f4"), ("imag", "f4")])
        dataset = f.create_dataset("complex_dataset", data=complex_data)
        dataset.attrs["MATLAB_class"] = np.string_("single")

    decoder = HDF5Decoder()
    with h5py.File(temp_mat_file, "r") as f:
        result = decoder.convert_mat(f["complex_dataset"], flatten_keys=False)

    assert np.all(result == np.array([1.0 + 2.0j], dtype=np.complex64))


def test_load_matlab(temp_files):
    temp_mat_file, _ = temp_files
    data = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
    sio.savemat(temp_mat_file, {"top_level": data})

    result = load_matlab(temp_mat_file, "top_level")

    assert "a" in result
    assert "b" in result
    np.testing.assert_array_equal(result["a"], [1, 2, 3])
    np.testing.assert_array_equal(result["b"], [4, 5, 6])


if __name__ == "__main__":
    pytest.main()
