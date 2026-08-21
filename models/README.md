# Models

The files below are hosted on Zenodo (**DOI TBD — added upon publication**)
rather than committed to this repository, since they're too large for git.
Download them and place them directly in this directory before running
`run_synthesis.py` or the figure scripts.

```
models/
├── natsounds_fmri_encoding_model_stimuli_first_rep_avg_robust.h5     (~867MB)
├── natsounds_fmri_encoding_model_stimuli_first_rep_avg_standard.h5   (~867MB)
├── robust_model_checkpoint.pt                                       (~225MB)
└── standard_model_checkpoint.pt                                     (~225MB)
```

The two `*_checkpoint.pt` files are the CochResNet50 weights (robust and
standard variants) that `scripts/cochresnet50/model.py` loads.
The two `*_encoding_model_*.h5` files are fitted encoding models — ridge
regression weights mapping CochResNet50 features to fMRI responses — saved
via `lib.encoding.save_h5` (see [`scripts/README.md`](../scripts/README.md#data-format)).
Every dataset/group inside these `.h5` files carries a `description` HDF5
attribute explaining its contents (e.g. `h5py.File(path)['coefs'].attrs['description']`).
