import os
from collections.abc import Iterable

import h5py
import numpy as np
from scipy import io as sio


class HDF5Decoder:
    """Modified based on https://github.com/skjerns/mat7.3/blob/master/mat73/__init__.py"""

    def __init__(self, load_only_su=True):
        self.load_only_su = load_only_su
        self._dict_class = dict
        self.refs = {}  # this is used in case of matlab matrices

    def mat2dict(self, hdf5):
        if "#refs#" in hdf5:
            self.refs = hdf5["#refs#"]
        d = self._dict_class()
        for var in hdf5:
            if var in ["#refs#", "#subsystem#"]:
                continue
            ext = os.path.splitext(hdf5.filename)[1].lower()
            if ext.lower() == ".mat":
                d[var] = self.unpack_mat(hdf5[var])
            elif ext == ".h5" or ext == ".hdf5":
                err = (
                    "Can only load .mat. Please use package hdfdict instead"
                    "\npip install hdfdict\n"
                    "https://github.com/SiggiGue/hdfdict"
                )
                raise NotImplementedError(err)
            else:
                raise ValueError("can only unpack .mat")
        return d

    def unpack_mat(self, hdf5, depth=0, flatten_keys=None):
        """
        unpack a h5py entry: if it's a group expand,
        if it's a dataset convert

        for safety reasons, the depth cannot be larger than 99
        """
        if depth == 99:
            raise RecursionError("Maximum number of 99 recursions reached.")
        if isinstance(hdf5, (h5py._hl.group.Group)):
            d = self._dict_class()
            if self.load_only_su and "SU" in hdf5:
                keys = ["SU"]
                if "StartDateTime" in hdf5:
                    keys.extend(["StartDateTime", "TimeZone"])
            else:
                keys = hdf5.keys()
            for key in keys:
                if key == "Fs":
                    # Flatten the values for the sampling rate. This is ugly, but what can you do.
                    flatten_keys = True
                matlab_class = hdf5[key].attrs.get("MATLAB_class")
                elem = hdf5[key]
                unpacked = self.unpack_mat(
                    elem, depth=depth + 1, flatten_keys=flatten_keys
                )
                if matlab_class == b"struct" and len(elem) > 1:
                    values = unpacked.values()

                    # we can only pack them together in MATLAB style if
                    # all subitems are the same lengths.
                    # MATLAB is a bit confusing here, and I hope this is
                    # correct. see https://github.com/skjerns/mat7.3/issues/6
                    allist = all([isinstance(item, list) for item in values])
                    if allist:
                        same_len = len(set([len(item) for item in values])) == 1
                    else:
                        same_len = False

                    # convert struct to its proper form as in MATLAB
                    # i.e. struct[0]['key'] will access the elements
                    # we only recreate the MATLAB style struct
                    # if all the subelements have the same length
                    # and are of type list
                    if allist and same_len:
                        items = list(zip(*[v for v in values]))

                        keys = unpacked.keys()
                        struct = [{k: v for v, k in zip(row, keys)} for row in items]
                        struct = [self._dict_class(d) for d in struct]
                        unpacked = struct
                d[key] = unpacked

            return d
        elif isinstance(hdf5, h5py._hl.dataset.Dataset):
            return self.convert_mat(hdf5, flatten_keys=flatten_keys)
        else:
            raise Exception(f"Unknown hdf5 type: {key}:{type(hdf5)}")

    def _has_refs(self, dataset, flatten_keys=False):
        if len(dataset) == 0:
            return False
        if not isinstance(dataset[0], np.ndarray):
            return False
        if isinstance(dataset[0][0], h5py.h5r.Reference):
            return True
        return False

    def convert_mat(self, dataset, flatten_keys):
        """
        Converts h5py.dataset into python native datatypes
        according to the matlab class annotation
        """
        # all MATLAB variables have the attribute MATLAB_class
        # if this is not present, it is not convertible
        if not "MATLAB_class" in dataset.attrs and not self._has_refs(dataset):
            return None

        if self._has_refs(dataset):
            mtype = "cell"
        else:
            mtype = dataset.attrs["MATLAB_class"].decode()

        if mtype == "cell":
            cell = []
            for ref in dataset:
                row = []
                # some weird style MATLAB have no refs, but direct floats or int
                if isinstance(ref, Iterable):
                    for r in ref:
                        entry = self.unpack_mat(self.refs.get(r))
                        row.append(entry)
                else:
                    row = [ref]
                cell.append(row)
            cell = list(map(list, zip(*cell)))  # transpose cell
            if len(cell) == 1:  # singular cells are interpreted as int/float
                cell = cell[0]
            return cell

        elif mtype == "char":
            string_array = np.array(dataset).ravel()
            string_array = "".join([chr(x) for x in string_array])
            string_array = string_array.replace("\x00", "")
            return string_array

        elif mtype == "bool":
            return bool(dataset)

        elif mtype == "logical":
            arr = np.array(dataset, dtype=bool).T.squeeze()
            if arr.size == 1:
                arr = bool(arr)
            return arr

        elif mtype == "canonical empty":
            return None

        # complex numbers need to be filtered out separately
        elif "imag" in str(dataset.dtype):
            if dataset.attrs["MATLAB_class"] == b"single":
                dtype = np.complex64
            else:
                dtype = np.complex128
            arr = np.array(dataset)
            arr = (arr["real"] + arr["imag"] * 1j).astype(dtype)
            return arr.T.squeeze()

        # if it is none of the above, we can convert to numpy array
        elif mtype in (
            "double",
            "single",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            arr = np.array(dataset, dtype=dataset.dtype)
            arr = arr.T.squeeze()
            if flatten_keys and arr.shape == tuple():
                return float(arr)
            return arr
        else:
            return None


def convert_mat_to_dict(data, load_only_su=True):
    """Convert the data from the matlab struct into a python dict."""
    out = {}
    for key in data._fieldnames:
        val = getattr(data, key)
        if isinstance(val, sio.matlab.mat_struct):
            if load_only_su and "SU" in val._fieldnames:
                out[key] = {}
                out[key]["SU"] = convert_mat_to_dict(
                    getattr(val, "SU"), load_only_su=load_only_su
                )
                if "StartDateTime" in val._fieldnames:
                    out[key]["StartDateTime"] = getattr(val, "StartDateTime")
                if "TimeZone" in val._fieldnames:
                    out[key]["TimeZone"] = getattr(val, "TimeZone")
            else:
                # recursive function call
                out[key] = convert_mat_to_dict(val, load_only_su=load_only_su)
        elif isinstance(val, (list, np.ndarray)) and len(val) == 0:
            out[key] = val
        elif isinstance(val, (list, np.ndarray)) and isinstance(
            val[0], sio.matlab.mat_struct
        ):
            tmp = [convert_mat_to_dict(v, load_only_su=load_only_su) for v in val]
            out[key] = tmp
        else:
            out[key] = val
    return out


def load_matlab(file_name, top_level, load_only_su=False):
    try:
        data = sio.loadmat(
            file_name, squeeze_me=True, struct_as_record=False, mat_dtype=True
        )
        data = data[top_level]
        data = convert_mat_to_dict(data, load_only_su=load_only_su)
    except NotImplementedError:
        # Matlab 7.3
        decoder = HDF5Decoder(load_only_su=load_only_su)
        with h5py.File(file_name, "r") as f:
            data = decoder.mat2dict(f)
            data = data[top_level]
    return data
