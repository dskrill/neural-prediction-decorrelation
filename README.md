# Neural Prediction Decorrelation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

Code and data for: **Neural prediction decorrelation reveals that adversarial robustness substantially improves DNN prediction accuracy across the entire human auditory cortex**

David Skrill<sup>1*</sup>, Jenelle Feather<sup>2,3</sup>, and Sam V. Norman-Haignere<sup>1,4-6</sup>

<sup>1</sup>Department of Biostatistics and Computational Biology, University of Rochester Medical Center
<sup>2</sup>Neuroscience Institute, Carnegie Mellon University
<sup>3</sup>Psychology Department, Carnegie Mellon University
<sup>4</sup>Department of Neuroscience, University of Rochester Medical Center
<sup>5</sup>Department of Brain and Cognitive Sciences, University of Rochester
<sup>6</sup>Department of Biomedical Engineering, University of Rochester

<sup>*</sup>Corresponding author: [david_skrill@urmc.rochester.edu](mailto:david_skrill@urmc.rochester.edu)

---

## Overview

This repository contains the codebase and data for our manuscript investigating how adversarial robustness in deep neural networks (DNNs) improves prediction accuracy of fMRI responses across human auditory cortex.

We introduce **neural prediction decorrelation (NPD)**, a method for synthesizing stimuli that decorrelate encoding model. Using fMRI, we show that adversarial robustness substantially improves prediction accuracy across the entire auditory cortex.

 >[!IMPORTANT] 
 > If you want to try NPD synthesis for yourself or get a feel for the method, check out `demo.ipynb`! This notebook is a mininal working example of NPD synthesis, and includes a demonstration that the sounds synthesized in the notebook successfully decorrelate neural predictions.
 >

## Repository Structure

```
neural-prediction-decorrelation/
├── scripts/              # Core library and executable scripts
│   ├── bin/               # Command-line executables
│   ├── lib/                # Python library (data, encoding, synthesis, analysis, utils)
│   ├── cochresnet50/       # CochResNet50 feature extraction model
│   └── README.md           # Detailed usage documentation
│
├── data/                  # Derived fMRI data (see Data Availability below)
│   ├── original-subjects/  # NH2015 & B2021 datasets (10+20 subjects)
│   └── new-subjects-clean/ # S2026 dataset (9 subjects)
│
├── models/                # Trained encoding models and CochResNet50 checkpoints
│
├── figures/               # Scripts for the manuscript's main figures
│
├── stimuli/               # Experimental stimuli (34MB)
│   ├── NPD/                # NPD-synthesized sounds
│   └── natural_sounds/     # Natural sound reference set
│
├── masks/                 # ROI and reliability masks
│
├── components/            # Stimulus/component metadata used by the figure scripts
│
└── README.md              
```

See [`scripts/README.md`](scripts/README.md) for detailed documentation of the codebase organization and usage.

---

## Installation

First, install the required system libraries, `ffmpeg`, `sox`, and `libsndfile1`, e.g.,

```bash
apt-get install ffmpeg sox libsndfile1
```

Then install the python libraries:

```bash
pip install -r requirements.txt
```
---

## Quick Start

### 1. Train an Encoding Model

```bash
python scripts/bin/train_encoding_model.py \
    --data data/original-subjects/natsounds.h5 \
    --n_pcs 80 \
    --stimuli first \
    --robust \
    --output models/robust_model.h5
```

### 2. Generate NPD Sounds

```bash
python scripts/bin/run_synthesis.py \
    --variance_multiplier 5 \
    --training_set_10 \
    --output_dir output
```

### 3. Load and Analyze Data

```python
from lib.data import load_data, get_subject_data
from lib.utils import mycorr

# Load predictions and observations
data = load_data('data/new-subjects-clean/observed_and_predicted_data_new_subjects_smoothed_stratified.npz')

# Extract subject data
obs = get_subject_data(data, 'sub-MRI004', 'observed', hemisphere='L')
pred_robust = get_subject_data(data, 'sub-MRI004', 'predictions', 'robust', 'L')

# Compute prediction accuracy
accuracy = mycorr(pred_robust, obs)
print(f"Prediction accuracy: {accuracy.mean():.3f}")
```

### 4. Create Figures

```bash
cd figures
python create_figure1.py --data-dir ../data --output-dir output_figure1
```

See [`scripts/README.md`](scripts/README.md) for more usage examples.

---

## Data Availability

Data from the following three experiments were used in this work:

1. **NH2015**: 10 subjects, natural sound stimuli (Norman-Haignere et al., 2015)
2. **B2021**: 20 subjects, natural sound stimuli (Boebinger et al., 2021)
3. **S2026**: 9 subjects, NPD and natural sounds (this study)

We make the data available in a pre-processed and collated format that is ready for downstream analyses. The raw scans are not currently available.

The derived data files referenced throughout this repo (`data/`, `masks/`,
`models/`, `components/`) are hosted on Zenodo at **[DOI TBD — added upon
publication]**, since they are too large for git. Download them into the
directory layout shown above before running any of the scripts.

All fMRI data were preprocessed using [fMRIPrep](https://fmriprep.org/).

For questions about the data, please contact the corresponding author.

---

## Reproducibility

### Reproducing Figures

The manuscript's main figures can be reproduced using the scripts in the `figures/` directory:

```bash
cd figures

python create_figure1.py               # Prediction accuracy on natural sounds
python create_figure2.py               # Decorrelated predictions using NPD
python create_figure3.py               # Prediction accuracy in new participants
python create_figure4_consolidated.py  # Analysis of low-dimensional representation
```
---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## System Requirements

### Hardware

- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 64GB+
- **GPU**: NVIDIA GPUs with CUDA support: one large (80 GB) GPU or two smaller (40 GB) GPUs

### Software

- **OS**: Linux (tested on Ubuntu 26.04+ and RHEL 9)
- **Python**: 3.11
- **CUDA**: 12.6+

---

## Contributing

This repository is primarily for reproducing published research. For questions or suggestions, contact the corresponding author: [david_skrill@urmc.rochester.edu](mailto:david_skrill@urmc.rochester.edu).

---

## Contact

**Corresponding Author**:
David Skrill, Ph.D.
Department of Biostatistics and Computational Biology
University of Rochester Medical Center
[david_skrill@urmc.rochester.edu](mailto:david_skrill@urmc.rochester.edu)

**Co-Authors**:
Jenelle Feather, Ph.D. ([jfeather@cmu.edu](mailto:jfeather@cmu.edu))
Sam V. Norman-Haignere, Ph.D. ([samuel_norman-haignere@urmc.rochester.edu](mailto:samuel_norman-haignere@urmc.rochester.edu))


**Last Updated**: August 2026
