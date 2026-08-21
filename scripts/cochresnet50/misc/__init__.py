"""Miscellaneous CochResNet50 utilities.

This subpackage contains the original model components (ResNet backbone,
audio transforms, adversarial attacker utilities, etc.) that were inherited
from the upstream codebase.

To maintain compatibility with existing serialized models (pickled with
module names like ``custom_modules`` or ``audio_functions``), we expose
those modules both under the ``cochresnet50.misc`` package namespace and
as top-level module aliases in ``sys.modules``.
"""

import sys

from . import audio_functions
from . import audio_input_representations
from . import custom_modules
from . import padding
from . import resnet
from . import attack_steps
from . import attacker
import chcochleagram.envelope_extraction as _env_extraction
import chcochleagram.cochlear_filters as _coch_filters
import chcochleagram.downsampling as _downsampling
import chcochleagram.cochleagram as _cochleagram
import chcochleagram.compression as _compression
import chcochleagram.helpers as _helpers

# ---------------------------------------------------------------------------
# Backward-compatibility aliases for pickled models
# ---------------------------------------------------------------------------
# Many legacy checkpoints were pickled when these modules lived at the
# top level (e.g., ``custom_modules`` instead of
# ``cochresnet50.misc.custom_modules``). To make unpickling work without
# modifying the checkpoints, we register aliases in ``sys.modules`` so that
# imports like ``import custom_modules`` succeed and resolve to the
# corresponding modules inside this package.

sys.modules.setdefault("audio_functions", audio_functions)
sys.modules.setdefault("audio_input_representations", audio_input_representations)
sys.modules.setdefault("custom_modules", custom_modules)
sys.modules.setdefault("padding", padding)
sys.modules.setdefault("resnet", resnet)
sys.modules.setdefault("attack_steps", attack_steps)
sys.modules.setdefault("attacker", attacker)
sys.modules.setdefault("envelope_extraction", _env_extraction)
sys.modules.setdefault("cochlear_filters", _coch_filters)
sys.modules.setdefault("downsampling", _downsampling)
sys.modules.setdefault("cochleagram", _cochleagram)
sys.modules.setdefault("compression", _compression)
sys.modules.setdefault("helpers", _helpers)
