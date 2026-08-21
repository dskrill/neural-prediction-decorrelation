# Data

The files below are hosted on Zenodo (**DOI TBD — added upon publication**)
rather than committed to this repository, since they're too large for git.
Download them and place them at the paths shown before running any scripts.

```
data/
├── original-subjects/
│   ├── natsounds.h5                                              (~1.8GB)
│   └── observed_and_predicted_data_original_subjects_hemi.npz    (~321MB)
│
└── new-subjects-clean/
    ├── first_stage_output_smooth3mm.h5                                           (~12GB)
    └── observed_and_predicted_data_new_subjects_smoothed_stratified.npz          (~23GB)
```

`first_stage_output_smooth3mm.h5` holds first-stage GLM response coefficients
(`coefs_L`/`coefs_R`, per subject/session, sliced to the 180 stimulus-related
regressors — the 80 nuisance/confound regressor columns in the original
first-stage output aren't used downstream and were dropped), plus the
subject IDs and ordered stimulus names.

See the top-level [README](../README.md#data-availability) for what these
datasets contain and how they were collected.
