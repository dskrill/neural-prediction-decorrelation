"""
Generic, portable HDF5 serialization for plain data dicts (pickled dicts of
numpy arrays, strings, and scalars, with no custom classes involved).

Used for natsounds.pkl / natsounds_v2.pkl / components/*.pkl / ROI_masks.pkl
-- files that only ever held dicts, lists, numpy arrays, and strings, so a
round-trip through pickle wasn't buying anything except a dependency on
Python's pickle protocol. HDF5 is readable from any language/version.

save_dict_h5(d, path) / load_dict_h5(path) round-trip an arbitrarily nested
structure of dict / list-of-arrays (ragged, i.e. different shapes) /
numpy arrays (numeric or string) / python scalars. On load, groups whose
keys are exactly '0', '1', ... come back as plain Python lists (so
`natsounds['original']['grid_x'][q]`-style integer indexing keeps working
unchanged); everything else round-trips as a dict.
"""
import h5py
import numpy as np


def _is_object_array_of_arrays(v):
    return (
        isinstance(v, np.ndarray)
        and v.dtype == object
        and len(v) > 0
        and isinstance(v[0], np.ndarray)
    )


def _is_string_like_array(v):
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("U", "S"):
            return True
        if v.dtype == object and len(v) > 0 and isinstance(v[0], str):
            return True
    return False


def _save_value(group, key, v):
    if isinstance(v, dict):
        sg = group.create_group(key)
        for k, vv in v.items():
            _save_value(sg, str(k), vv)
    elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
        group.create_dataset(key, data=np.asarray(v, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
    elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
        # Ragged list of arrays (e.g. per-hemisphere grids of different shape)
        sg = group.create_group(key)
        for i, item in enumerate(v):
            _save_value(sg, str(i), item)
    elif _is_object_array_of_arrays(v):
        sg = group.create_group(key)
        for i, item in enumerate(v):
            _save_value(sg, str(i), item)
    elif _is_string_like_array(v):
        group.create_dataset(key, data=np.asarray(v, dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
    elif isinstance(v, np.ndarray):
        group.create_dataset(key, data=v)
    elif isinstance(v, (list, tuple)):
        group.create_dataset(key, data=np.asarray(v))
    elif isinstance(v, str):
        group.attrs[key] = v
    elif isinstance(v, (bool, int, float, np.integer, np.floating, np.bool_)):
        group.attrs[key] = v
    else:
        raise TypeError(f"Don't know how to save key {key!r} of type {type(v)}")


def save_dict_h5(d: dict, path: str) -> None:
    """Save a nested dict of arrays/lists/scalars to a portable HDF5 file."""
    with h5py.File(path, "w") as f:
        for k, v in d.items():
            _save_value(f, str(k), v)


def _load_group(group):
    keys = list(group.keys()) + list(group.attrs.keys())
    is_indexed_list = (
        len(group.keys()) > 0
        and all(k.isdigit() for k in group.keys())
        and len(group.attrs.keys()) == 0
    )

    if is_indexed_list:
        return [_load_item(group[str(i)]) for i in range(len(group.keys()))]

    out = {}
    for k in group.keys():
        out[k] = _load_item(group[k])
    for k, v in group.attrs.items():
        out[k] = v
    return out


def _load_item(item):
    if isinstance(item, h5py.Group):
        return _load_group(item)
    data = item[()]
    if isinstance(data, bytes):
        return data.decode("utf-8")
    if isinstance(data, np.ndarray) and data.dtype == object:
        # string_dtype datasets decode to bytes objects; normalize to str
        return np.array(
            [x.decode("utf-8") if isinstance(x, bytes) else x for x in data],
            dtype=object,
        )
    return data


def load_dict_h5(path: str) -> dict:
    """Reconstruct the nested dict/list/array structure saved by save_dict_h5."""
    with h5py.File(path, "r") as f:
        return _load_group(f)
