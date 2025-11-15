"""
ARC-AGI-1 Dataset Loader and Augmentation

A clean, foundational codebase for loading and augmenting ARC-AGI-1 puzzles.
"""

from .loader import ARCDataset, ARCPuzzle, load_arc_dataset
from .augmentation import (
    AugmentationConfig,
    augment_grid,
    color_shuffle,
    rotate_grid,
    reflect_grid,
    translate_grid,
)
from .config import DatasetConfig

__version__ = "0.1.0"

__all__ = [
    # Loader
    "ARCDataset",
    "ARCPuzzle",
    "load_arc_dataset",
    # Augmentation
    "AugmentationConfig",
    "augment_grid",
    "color_shuffle",
    "rotate_grid",
    "reflect_grid",
    "translate_grid",
    # Config
    "DatasetConfig",
]
