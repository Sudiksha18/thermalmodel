"""Euclidean Technologies Thermal Anomaly Detection System"""

__version__ = "1.0.0"
__author__ = "Euclidean Technologies"
__email__ = "contact@euclideantech.ai"

from . import dataloader
from . import models
from . import train
from . import inference
from . import utils
from . import evaluation

__all__ = [
    "dataloader",
    "models", 
    "train",
    "inference",
    "utils",
    "evaluation"
]