#!/usr/bin/env python
"""
Generate cochleagram visualizations for all stimuli in dataset.

This script loads stimulus waveforms and creates cochleagram representations
using the chcochleagram library, saving both labeled and clean versions.
"""
import os
import sys
sys.path.append("../scripts")

import chcochleagram
import chcochleagram.cochlear_filters
import chcochleagram.envelope_extraction
import chcochleagram.downsampling
import chcochleagram.compression
import chcochleagram.cochleagram
import soundfile as sf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

plt.rcParams['svg.fonttype'] = 'none'
# set font to size 10 Times New Roman
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Times New Roman'

# Import data loading function from lib
from lib.data import load_data

data = load_data(
        filename=str('../data/new-subjects-clean/observed_and_predicted_data_new_subjects_smoothed_stratified.npz'),
        components=str('../components/components_reordered_stratified.h5'),
    )

### Args used for multiple stages of the cochleagram operations
signal_size = 40000 # Length of the input audio signal (currently must be fixed, due to filter construction)
sr = 20000 # Sampling rate of the input audio
pad_factor = 1.25 # Zero padding applied to the waveform, so the end signal is length pad_factor*signal_length
use_rfft = True # Whether to use rfft operations when appropriate (recommended)

### Define the cochlear filters using ERBCosFilters. 
# These are the arguments used for filter construction of ERBCosFilters. See helpers/erb_filters.py for 
# more documentation. 
half_cos_filter_kwargs = {
    'n':50, # Number of filters to evenly tile the space
    'low_lim':50, # Lowest center frequency for full filter (if lowpass filters are used they can be centered lower)
    'high_lim':10000, # Highest center frequency 
    'sample_factor':4, # Positive integer that determines how densely ERB function will be sampled
    'full_filter':False, # Whether to use the full-filter. Must be False if rFFT is true. 
}
# These arguments are for the CochFilters class (generic to any filters). 
coch_filter_kwargs = {'use_rfft':use_rfft,
                      'pad_factor':pad_factor,
                      'filter_kwargs':half_cos_filter_kwargs}

# This (and most) cochleagrams use ERBCosFilters, however other types of filterbanks can be 
# constructed for linear spaced filters or different shapes. Make a new CochlearFilter class for 
# these. 
filters = chcochleagram.cochlear_filters.ERBCosFilters(signal_size,
                                                       sr, 
                                                       **coch_filter_kwargs)

### Look at the filters 
# Filters are in form [filter_idx, frequency, (real_value, complex_value)] where for ERB filters 
# the complex component is 0. Newer versions of pytorch have complex types, but complex values are 
# represented with an extra dimension for now. 

### Define an envelope extraction operation
# Use the analytic amplitude of the hilbert transform here. Other types of envelope extraction 
# are also implemented in envelope_extraction.py. Can use Identity if want the raw subbands. 
envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(signal_size,
                                                                                  sr, 
                                                                                  use_rfft, 
                                                                                  pad_factor)

### Define a downsampling operation
# Downsample the extracted envelopes. Can use Identity if want the raw subbands. 
env_sr = 200 # Sampling rate after downsampling
downsampling_kwargs = {'window_size':1001} # Parameters for the downsampling filter (see downsampling.py)
downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(sr, env_sr, **downsampling_kwargs)

### Define a compression operation.
compression_kwargs = {'power':0.3, # Power compression of 0.3 
                      'offset':1e-8, # Offset for numerical stability in backwards pass
                      'scale':1, # Optional multiplicative value applied to the envelopes before compression 
                      'clip_value':100} # Clip the gradients for this compression for stability
compression = chcochleagram.compression.ClippedGradPowerCompression(**compression_kwargs)
### Once the operations are defined, put them all together into the Cochleagram module. 
cochleagram = chcochleagram.cochleagram.Cochleagram(filters, 
                                                    envelope_extraction,
                                                    downsampling_op,
                                                    compression=compression)

data['stimuli_waveforms'].keys()

for stim_name,waveform in tqdm(data['stimuli_waveforms'].items()):
    y = cochleagram(torch.from_numpy(waveform).unsqueeze(0))

    # Plot the cochleagram
    fig = plt.figure(figsize=(1.25, 1.25))
    ax = fig.add_subplot(111)
    ax.imshow(np.squeeze(y.detach().numpy()), origin='lower', extent=(0, y.shape[2], 0, y.shape[1]),interpolation='none',cmap="Blues")

    # plt.title('Cochleagram (with ERBFilters) for a sum of sine waves \n with frequencies %d, %d, and %d Hz'%(freq, freq2, freq3))

    ## Depending on the temporal padding the cochleagram length may not be exactly equal env_sr*signal_size/sr
    # Because of this, set the x-axis tick labels based on the original audio. 
    num_ticks = 2
    x_tick_numbers = [t_num*y.shape[-1]/(num_ticks-1) for t_num in range(num_ticks)]
    x_tick_labels = [t_num*signal_size/sr/(num_ticks-1) for t_num in range(num_ticks)]
    ax.set_xticks(x_tick_numbers)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel('Time (s)')
    # get rid of the stim{xxx} prefix where present
    if stim_name.startswith('stim'):
        stim_name_formatted = " ".join(stim_name.split("_")[1:]).title()
    else:
        stim_name_formatted = stim_name.replace("NPD_", "NPD #").replace("stim", "")
    ax.set_title(stim_name_formatted,wrap=True)

    ## Label the frequency axis based on the center frequencies for the ERB filters. 
    filters.filter_extras['cf']
    # Use ticks starting at the lowest non-lowpass filter center frequency. 
    # y_ticks = [y_t+3 for y_t in [plt.yticks()[0][0],plt.yticks()[0][-1]] if y_t<=y.shape[1]]
    # ax.set_yticks(y_ticks, [int(round(filters.filter_extras['cf'][int(f_num)])) for f_num in y_ticks])
    ax.set_yticks([3,y.shape[1]-3],[50,'10k'])
    ax.set_ylabel('Frequency (Hz)')
    # fig.savefig(f"stim_cochleagrams/cochleagram_{stim_name}.svg",format='svg')
    fig.savefig(f"stim_cochleagrams/cochleagram_{stim_name}.png",format='png',bbox_inches='tight',dpi=1200)
    plt.close()

    # Also save clean versions, without any text whatsoever but with tick marks
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    # fig.savefig(f"stim_cochleagrams/cochleagram_{stim_name}_clean.svg",format='svg')
    fig.savefig(f"stim_cochleagrams/cochleagram_{stim_name}_clean.png",format='png',bbox_inches='tight',dpi=1200)
    plt.close()

    # save the waveform
    sf.write(f"stim_cochleagrams/{stim_name}.wav",waveform,sr)
