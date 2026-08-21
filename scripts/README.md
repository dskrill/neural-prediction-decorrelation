# Scripts Directory Structure

This directory contains the codebase for training encoding models, running NPD
synthesis, and generating the manuscript's main figures.

## Directory Structure

```
scripts/
├── bin/                   # Executable, mostly stand-alone scripts
│   ├── train_encoding_model.py                        # Train encoding models (NH2015 and B2021)
│   ├── fit_encoding_models_new_subjects_stratified.py # Train models (S2026, stratified train/test splits)
│   └── run_synthesis.py                               # Run NPD sound synthesis
│
├── lib/                   # Library code
│   ├── data/               # Data loading
│   │   ├── loaders.py             # Load NPZ data files
│   │   └── io_h5.py               # Portable HDF5 serialization for plain data dicts
│   │
│   ├── encoding/           # Encoding model functionality
│   │   ├── model.py               # EncodingModel class
│   │   ├── preprocessing.py       # Lag, StandardScaler
│   │   ├── ridge_utils.py         # Ridge regression utilities
│   │   └── io_h5.py               # Portable HDF5 save/load for fitted EncodingModel state
│   │
│   ├── synthesis/          # Sound synthesis
│   │   ├── synthesizer.py         # BestLayerSynthesizer classes
│   │   └── utils.py               # Synthesis utilities
│   │
│   ├── analysis/           # Analysis functions
│   │   └── reliability.py         # Split-half reliability
│   │
│   └── utils/               # General utilities
│       ├── audio.py               # Audio processing
│       ├── math.py                # Mathematical operations
│       └── visualization.py       # Brain visualization
│
└── cochresnet50/           # CochResNet50 model
    ├── model.py                   # Model instantiation
    └── misc/                      # Dependencies (inherited)
```

Figure-generation scripts (`create_figure1.py`, `create_figure2.py`,
`create_figure3.py`, `create_figure4_consolidated.py`, `make_cochleagrams.py`)
live under [`figures/`](../figures/) rather than here, and import this
`scripts/lib` package the same way the scripts below do.

## Data format

All archived data (`data/`, `masks/`, `components/`, `models/`) is stored as
HDF5 (`.h5`) or NumPy (`.npy`/`.npz`), not pickle — this keeps everything
readable regardless of Python/library version, and from languages other than
Python. Use `lib.data.load_dict_h5` / `lib.data.save_dict_h5` for plain data
dicts, and `lib.encoding.load_h5` / `lib.encoding.save_h5` for fitted
`EncodingModel` objects specifically (these reconstruct a live model via its
normal constructor rather than unpickling a frozen class instance).

## Usage Examples

### Training an Encoding Model

```python
from lib.encoding import EncodingModel, layers, save_h5

model = EncodingModel(
    robust=True,
    n_pcs=80,
    layers=layers,
    device='cuda'
)
model.fit(D_train, X_train, D_test, X_test)
save_h5(model, 'my_model.h5')
```

### Loading CochResNet50

```python
from cochresnet50 import instantiate_cochresnet50

model = instantiate_cochresnet50(robust=True, duration=2)
```

### Running Synthesis

```python
from lib.synthesis import BestLayerSynthesizer
from lib.encoding import load_h5

# Load models
robust_model = load_h5('robust.h5')
standard_model = load_h5('standard.h5')

# Create synthesizer
synth = BestLayerSynthesizer(
    encoding_models={
        'robust': robust_model,
        'standard': standard_model
    },
    batch_size=60,
    max_iters=10000,
    device='cuda'
)

# Run synthesis
synth.fit(train_indices=train_mask, test_indices=test_mask)
```

### Loading Data

```python
from lib.data import load_data, get_subject_data

# Load NPZ data
data = load_data('data/new-subjects-clean/observed_and_predicted_data_new_subjects_smoothed_stratified.npz')

# Get subject-specific data
obs = get_subject_data(data, 'sub-MRI004', 'observed', hemisphere='L')
pred = get_subject_data(data, 'sub-MRI004', 'predictions', 'robust', 'L')
```

### Using Math & Audio Utilities

```python
from lib.utils import mycorr, mycov, rms_normalize, CustomFade

# Compute correlation
corr = mycorr(predictions, observations)

# Normalize audio
audio = rms_normalize(waveform, rms_level=0.1)
```

### Brain Visualization

```python
from lib.utils import plot_brain, set_plot_area, color_lookup
import numpy as np

# Plot brain statistics
stat = np.random.randn(163842)
mask = stat > 0
plot_brain(stat, mask, fname='output/brain.png')

# Get category colors
speech_color = color_lookup['EngSpeech']
NPD_color = color_lookup['NPD']
```

## Command-Line Scripts

All scripts should be run from the project root using relative paths.

### Train Encoding Model (Original Subjects)
```bash
python scripts/bin/train_encoding_model.py \
    --data data/original-subjects/natsounds.h5 \
    --n_pcs 80 \
    --stimuli first \
    --robust \
    --output models/my_model.h5
```

### Train Encoding Model (New Subjects, Stratified)
```bash
# Group-level model
python scripts/bin/fit_encoding_models_new_subjects_stratified.py \
    --model_type robust \
    --analysis_type group \
    --first-stage-output data/new-subjects-clean/first_stage_output_smooth3mm.h5 \
    --output-dir models

# Single subject model
python scripts/bin/fit_encoding_models_new_subjects_stratified.py \
    --model_type robust \
    --analysis_type single_subject \
    --subject_id sub-MRI004 \
    --output-dir models
```

### Run Synthesis
```bash
python scripts/bin/run_synthesis.py \
    --variance_multiplier 5 \
    --training_set_10 \
    --output_dir output
```

### Generate Cochleagrams
```bash
cd figures
python make_cochleagrams.py
# Creates cochleagram visualizations for all stimuli, saved to stim_cochleagrams/
```
