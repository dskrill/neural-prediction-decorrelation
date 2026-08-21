"""
CochResNet50 model loading and instantiation.

This module provides the interface for loading pretrained CochResNet50 models
(both robust and standard versions).
"""
from .model import instantiate_cochresnet50

__all__ = ['instantiate_cochresnet50']

