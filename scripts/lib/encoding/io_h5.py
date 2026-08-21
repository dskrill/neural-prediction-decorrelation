"""
Portable HDF5 serialization for fitted EncodingModel state.

Unlike pickle/dill, this format doesn't depend on Python's object model or
on the exact class layout of EncodingModel/StandardScaler/Lag staying fixed
over time. It stores only plain arrays and scalars, and reconstructs a live
EncodingModel by calling its normal constructor (which rebuilds the
CochResNet50 backbone from the checkpoint files) and then repopulating the
fitted attributes.
"""
import h5py
import numpy as np
import torch

from .model import EncodingModel
from .preprocessing import StandardScaler

# Attributes computed by EncodingModel.fit() that need to be archived.
# (self.model / self.activations / self.lag are excluded: model is rebuilt
# by the constructor from the checkpoint files, activations is transient,
# and lag is fully determined by n_lags.)
_LAYERWISE_DICT_ATTRS = [
    "coefs",
    "intercepts",
    "all_predictions",
    "layerwise_train_predictions",
    "layerwise_train_correlations",
    "layerwise_train_MSEs",
    "layerwise_test_predictions",
    "layerwise_test_correlations",
    "layerwise_test_MSEs",
]
_ARRAY_ATTRS = ["best_predictions", "corrs_test", "MSE_test"]

_NONE_SENTINEL = "__none__"


def save_h5(model: EncodingModel, path: str) -> None:
    """Save a fitted EncodingModel's state to a portable HDF5 file."""
    with h5py.File(path, "w") as f:
        f.attrs["model_name"] = model.model_name
        f.attrs["robust"] = bool(model.robust)
        f.attrs["n_pcs"] = model.n_pcs
        f.attrs["n_lags"] = model.n_lags
        f.attrs["standardize_features"] = bool(model.standardize_features)
        f.attrs["device"] = "cpu"  # archived state is always CPU-side

        g = f.create_group("ridge_kwargs")
        for k, v in model.ridge_kwargs.items():
            if isinstance(v, np.ndarray):
                g.create_dataset(k, data=v)
            else:
                g.attrs[k] = v

        g = f.create_group("layers")
        for k, v in model.layers.items():
            g.attrs[k] = v

        g = f.create_group("scalers")
        for layer, scaler in model.scalers.items():
            sg = g.create_group(layer)
            sg.create_dataset("mean", data=scaler.mean.detach().cpu().numpy())
            sg.create_dataset("std", data=scaler.std.detach().cpu().numpy())
            sg.attrs["epsilon"] = scaler.epsilon
            sg.attrs["dim"] = _NONE_SENTINEL if scaler.dim is None else scaler.dim
            sg.attrs["use_norm"] = bool(scaler.use_norm)
            sg.attrs["mean_dim"] = (
                _NONE_SENTINEL if scaler.mean_dim is None else scaler.mean_dim
            )
            sg.attrs["std_dim"] = (
                _NONE_SENTINEL if scaler.std_dim is None else scaler.std_dim
            )

        g = f.create_group("V")
        for layer, v in model.V.items():
            g.create_dataset(layer, data=v.detach().cpu().numpy())

        for attr in _LAYERWISE_DICT_ATTRS:
            g = f.create_group(attr)
            for layer, arr in getattr(model, attr).items():
                g.create_dataset(layer, data=np.asarray(arr))

        for attr in _ARRAY_ATTRS:
            f.create_dataset(attr, data=np.asarray(getattr(model, attr)))

        # best_layers: dict[int voxel_index -> str layer name] -> string array
        n_voxels = len(model.best_layers)
        best_layers_arr = np.array(
            [model.best_layers[i] for i in range(n_voxels)], dtype="S32"
        )
        f.create_dataset("best_layers", data=best_layers_arr)


def load_h5(path: str, device: str = "cpu") -> EncodingModel:
    """Reconstruct a live EncodingModel from an archive written by save_h5.

    Rebuilds the CochResNet50 backbone via the normal constructor (reading
    the current checkpoint files), then repopulates the fitted state.
    """
    with h5py.File(path, "r") as f:
        ridge_kwargs = {}
        rg = f["ridge_kwargs"]
        for k in rg.keys():
            ridge_kwargs[k] = rg[k][()]
        for k, v in rg.attrs.items():
            ridge_kwargs[k] = v

        layers = {k: v for k, v in f["layers"].attrs.items()}

        model = EncodingModel(
            model_name=f.attrs["model_name"],
            robust=bool(f.attrs["robust"]),
            n_pcs=int(f.attrs["n_pcs"]),
            n_lags=int(f.attrs["n_lags"]),
            ridge_kwargs=ridge_kwargs,
            layers=layers,
            standardize_features=bool(f.attrs["standardize_features"]),
            device=device,
        )

        scalers = {}
        sg = f["scalers"]
        for layer in sg.keys():
            s = sg[layer]

            def _unsentinel(v):
                return None if v == _NONE_SENTINEL else v

            scalers[layer] = StandardScaler(
                mean=torch.from_numpy(s["mean"][()]).to(device),
                std=torch.from_numpy(s["std"][()]).to(device),
                epsilon=float(s.attrs["epsilon"]),
                dim=_unsentinel(s.attrs["dim"]),
                use_norm=bool(s.attrs["use_norm"]),
                mean_dim=_unsentinel(s.attrs["mean_dim"]),
                std_dim=_unsentinel(s.attrs["std_dim"]),
            )
        model.scalers = scalers

        model.V = {
            layer: torch.from_numpy(f["V"][layer][()]).to(device)
            for layer in f["V"].keys()
        }

        for attr in _LAYERWISE_DICT_ATTRS:
            g = f[attr]
            setattr(model, attr, {layer: g[layer][()] for layer in g.keys()})

        for attr in _ARRAY_ATTRS:
            setattr(model, attr, f[attr][()])

        best_layers_arr = f["best_layers"][()]
        model.best_layers = {
            i: name.decode("utf-8") for i, name in enumerate(best_layers_arr)
        }

    return model
