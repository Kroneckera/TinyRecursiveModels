"""
Configuration for ARC-AGI-1 Dataset Processing

Centralized configuration for dataset loading, augmentation, and processing.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class DatasetConfig:
    """
    Complete configuration for ARC-AGI-1 dataset processing.

    Example:
        >>> config = DatasetConfig(
        ...     data_dir="kaggle/combined",
        ...     subsets=["training", "evaluation"],
        ...     num_augmentations=1000,
        ...     target_grid_size=(30, 30)
        ... )
    """
    # Data paths
    data_dir: str = "kaggle/combined"
    subsets: List[str] = field(default_factory=lambda: ["training"])

    # Augmentation settings
    num_augmentations: int = 1000
    enable_color_shuffle: bool = True
    enable_rotation: bool = True
    enable_reflection: bool = True
    enable_translation: bool = True
    color_shuffle_keep_black: bool = True

    # Grid size settings
    target_grid_size: Optional[Tuple[int, int]] = (30, 30)
    max_grid_size: int = 30

    # Random seed
    seed: int = 42

    # Processing options
    max_puzzles: Optional[int] = None  # Limit number of puzzles (for debugging)
    deduplication: bool = True  # Remove duplicate augmented grids

    # Output settings
    output_dir: str = "data/arc1-processed"
    save_format: str = "npy"  # 'npy' or 'npz'

    def __post_init__(self):
        """Validate configuration."""
        # Validate subsets
        valid_subsets = {"training", "evaluation", "concept"}
        for subset in self.subsets:
            if subset not in valid_subsets:
                raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_subsets}")

        # Validate target grid size
        if self.target_grid_size is not None:
            h, w = self.target_grid_size
            if h <= 0 or w <= 0:
                raise ValueError(f"Target grid size must be positive, got {self.target_grid_size}")
            if h > self.max_grid_size or w > self.max_grid_size:
                raise ValueError(
                    f"Target grid size {self.target_grid_size} exceeds "
                    f"max size {self.max_grid_size}"
                )

        # Validate augmentation count
        if self.num_augmentations < 0:
            raise ValueError(f"num_augmentations must be non-negative, got {self.num_augmentations}")

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def get_challenges_path(self, subset: str) -> Path:
        """Get path to challenges file for given subset."""
        return Path(self.data_dir) / f"arc-agi_{subset}_challenges.json"

    def get_solutions_path(self, subset: str) -> Path:
        """Get path to solutions file for given subset."""
        return Path(self.data_dir) / f"arc-agi_{subset}_solutions.json"

    def get_output_path(self, subset: str, split: str = "train") -> Path:
        """
        Get output path for processed data.

        Args:
            subset: Dataset subset (e.g., 'training', 'evaluation')
            split: Data split ('train' or 'test')

        Returns:
            Path object
        """
        output_path = Path(self.output_dir) / subset / split
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data_dir': self.data_dir,
            'subsets': self.subsets,
            'num_augmentations': self.num_augmentations,
            'enable_color_shuffle': self.enable_color_shuffle,
            'enable_rotation': self.enable_rotation,
            'enable_reflection': self.enable_reflection,
            'enable_translation': self.enable_translation,
            'color_shuffle_keep_black': self.color_shuffle_keep_black,
            'target_grid_size': self.target_grid_size,
            'max_grid_size': self.max_grid_size,
            'seed': self.seed,
            'max_puzzles': self.max_puzzles,
            'deduplication': self.deduplication,
            'output_dir': self.output_dir,
            'save_format': self.save_format,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'DatasetConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# Preset configurations

def get_default_config() -> DatasetConfig:
    """Get default configuration for full ARC-AGI-1 processing."""
    return DatasetConfig(
        data_dir="kaggle/combined",
        subsets=["training", "evaluation", "concept"],
        num_augmentations=1000,
        target_grid_size=(30, 30),
        seed=42,
    )


def get_debug_config() -> DatasetConfig:
    """Get configuration for quick debugging (small dataset, few augmentations)."""
    return DatasetConfig(
        data_dir="kaggle/combined",
        subsets=["training"],
        num_augmentations=10,
        max_puzzles=5,
        target_grid_size=(30, 30),
        seed=42,
        output_dir="data/arc1-debug",
    )


def get_no_augmentation_config() -> DatasetConfig:
    """Get configuration with no augmentation (original data only)."""
    return DatasetConfig(
        data_dir="kaggle/combined",
        subsets=["training", "evaluation"],
        num_augmentations=0,
        enable_color_shuffle=False,
        enable_rotation=False,
        enable_reflection=False,
        enable_translation=False,
        target_grid_size=None,
        seed=42,
    )
