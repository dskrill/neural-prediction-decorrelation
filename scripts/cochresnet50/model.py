"""
CochResNet50 model instantiation.

This module wraps the instantiation of CochResNet50 models with pretrained weights.
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from misc subdirectory
from .misc import resnet
from .misc import audio_input_representations as air
from .misc import custom_modules
from .misc.attacker import AttackerModel


def instantiate_cochresnet50(robust=True, duration=2):
    """
    Instantiate a CochResNet50 model with pretrained weights.
    
    Args:
        robust (bool): If True, load robustly-trained model. If False, load standard model.
        duration (float): Duration of audio signals in seconds.
    
    Returns:
        torch.nn.Module: Pretrained CochResNet50 model in eval mode.
    """
    audio_representation = 'cochleagram_1'
    num_classes = 794

    # Defines cochleagram parameters
    AUDIO_REP_PROPERTIES = air.AUDIO_INPUT_REPRESENTATIONS[audio_representation]
    AUDIO_REP_PROPERTIES['rep_kwargs']['signal_size'] = int(20000 * duration)

    # Build model architecture
    classifier_model = custom_modules.SequentialAttacker(
        custom_modules.AudioInputRepresentation(**AUDIO_REP_PROPERTIES),
        resnet.resnet50(num_classes=num_classes)
    )

    # Load pretrained weights
    # Determine checkpoint path relative to this file
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    
    if robust:
        checkpoint_path = os.path.join(checkpoint_dir, 'robust_model_checkpoint.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'standard_model_checkpoint.pt')
    
    # Load checkpoint onto CPU first; the caller moves the model to its
    # target device afterward (via .to(device)), so this keeps checkpoint
    # loading working on machines without a CUDA device available.
    cp = torch.load(checkpoint_path, pickle_module=dill, map_location='cpu')
    cp_renamed = copy.deepcopy(cp['model'])
    
    # Rename keys to match model architecture
    if robust:
        cp_renamed = {
            x.replace("module.model.", "1.").replace("module.audio_rep_transform.", "0.full_rep."): v 
            for x, v in cp['model'].items() 
            if "attacker" not in x and "module.preproc" not in x
        }
    else:
        cp_renamed = {
            x.replace("module.model.", "").replace("module.audio_rep_transform.", "0.full_rep."): v 
            for x, v in cp['model'].items() 
            if "attacker" not in x and "module.preproc" not in x
        }

    # Replace cochleagram-related parts of state dict
    keys_to_replace = [key for key in cp_renamed.keys() if key.startswith("0.full_rep.rep.Cochleagram")]
    for key in keys_to_replace:
        cp_renamed[key] = classifier_model.state_dict()[key]
    
    # Load state dict and set to eval mode
    classifier_model.load_state_dict(cp_renamed)
    classifier_model.eval()

    return classifier_model

