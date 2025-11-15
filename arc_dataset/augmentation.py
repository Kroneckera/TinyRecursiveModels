"""
Grid Augmentation for ARC-AGI Puzzles

Provides various augmentation operations that preserve puzzle semantics:
- Color shuffling (permute colors 1-9, keep 0/black fixed)
- Rotation (90°, 180°, 270°)
- Reflection (horizontal, vertical, diagonal, anti-diagonal)
- Translation (random padding within target size)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RotationType(Enum):
    """Types of rotation."""
    ROTATE_90 = 1
    ROTATE_180 = 2
    ROTATE_270 = 3


class ReflectionType(Enum):
    """Types of reflection."""
    HORIZONTAL = 0  # Flip left-right
    VERTICAL = 1    # Flip up-down
    DIAGONAL = 2    # Flip along main diagonal (transpose)
    ANTI_DIAGONAL = 3  # Flip along anti-diagonal


@dataclass
class AugmentationConfig:
    """
    Configuration for grid augmentation.

    Attributes:
        enable_color_shuffle: Whether to shuffle colors
        enable_rotation: Whether to apply rotation
        enable_reflection: Whether to apply reflection
        enable_translation: Whether to apply random translation/padding
        target_size: Target grid size for padding (H, W). If None, no padding.
        color_shuffle_keep_black: If True, color 0 (black) is never shuffled
        seed: Random seed for reproducibility
    """
    enable_color_shuffle: bool = True
    enable_rotation: bool = True
    enable_reflection: bool = True
    enable_translation: bool = True
    target_size: Optional[Tuple[int, int]] = (30, 30)
    color_shuffle_keep_black: bool = True
    seed: Optional[int] = None


def color_shuffle(
    grid: np.ndarray,
    keep_black: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle colors in the grid.

    Args:
        grid: Input grid (H, W) with values 0-9
        keep_black: If True, color 0 is not shuffled
        rng: Random number generator

    Returns:
        Tuple of (augmented_grid, color_mapping)
        color_mapping is the permutation used (to reverse if needed)

    Example:
        >>> grid = np.array([[0, 1, 2], [3, 4, 5]])
        >>> aug_grid, mapping = color_shuffle(grid, keep_black=True)
        >>> # mapping might be [0, 3, 1, 5, 2, 9, 4, 7, 6, 8]
    """
    if rng is None:
        rng = np.random.default_rng()

    assert grid.ndim == 2, "Grid must be 2D"
    assert np.all((grid >= 0) & (grid <= 9)), "Grid values must be 0-9"

    # Create color mapping
    if keep_black:
        # Keep 0 fixed, shuffle 1-9
        mapping = np.zeros(10, dtype=np.uint8)
        mapping[0] = 0
        mapping[1:] = rng.permutation(np.arange(1, 10, dtype=np.uint8))
    else:
        # Shuffle all colors 0-9
        mapping = rng.permutation(np.arange(10, dtype=np.uint8))

    # Apply mapping
    augmented = mapping[grid]

    return augmented, mapping


def rotate_grid(grid: np.ndarray, rotation: RotationType) -> np.ndarray:
    """
    Rotate grid by 90°, 180°, or 270° clockwise.

    Args:
        grid: Input grid (H, W)
        rotation: Type of rotation

    Returns:
        Rotated grid

    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> rotated = rotate_grid(grid, RotationType.ROTATE_90)
        >>> # Result: [[3, 1], [4, 2]]
    """
    assert grid.ndim == 2, "Grid must be 2D"

    if rotation == RotationType.ROTATE_90:
        # Rotate 90° clockwise: transpose then flip horizontally
        return np.flip(grid.T, axis=1)
    elif rotation == RotationType.ROTATE_180:
        # Rotate 180°: flip both axes
        return np.flip(grid, axis=(0, 1))
    elif rotation == RotationType.ROTATE_270:
        # Rotate 270° clockwise (= 90° counter-clockwise): flip horizontally then transpose
        return np.flip(grid, axis=1).T
    else:
        raise ValueError(f"Unknown rotation type: {rotation}")


def reflect_grid(grid: np.ndarray, reflection: ReflectionType) -> np.ndarray:
    """
    Reflect grid along specified axis.

    Args:
        grid: Input grid (H, W)
        reflection: Type of reflection

    Returns:
        Reflected grid

    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> reflected = reflect_grid(grid, ReflectionType.HORIZONTAL)
        >>> # Result: [[2, 1], [4, 3]]
    """
    assert grid.ndim == 2, "Grid must be 2D"

    if reflection == ReflectionType.HORIZONTAL:
        # Flip left-right
        return np.flip(grid, axis=1)
    elif reflection == ReflectionType.VERTICAL:
        # Flip up-down
        return np.flip(grid, axis=0)
    elif reflection == ReflectionType.DIAGONAL:
        # Transpose (flip along main diagonal)
        return grid.T
    elif reflection == ReflectionType.ANTI_DIAGONAL:
        # Flip along anti-diagonal: transpose then flip both axes
        return np.flip(grid.T, axis=(0, 1))
    else:
        raise ValueError(f"Unknown reflection type: {reflection}")


def translate_grid(
    grid: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 0,
    random_position: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad grid to target size with optional random translation.

    Args:
        grid: Input grid (H, W)
        target_size: Target size (H_target, W_target)
        pad_value: Value to use for padding (default: 0 = black)
        random_position: If True, randomly position grid within target.
                        If False, position at top-left (0, 0).
        rng: Random number generator

    Returns:
        Tuple of (padded_grid, position)
        position is (row_offset, col_offset) where grid was placed

    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> padded, pos = translate_grid(grid, target_size=(5, 5))
        >>> # padded is 5×5 with grid randomly positioned
        >>> # pos might be (1, 2) meaning grid starts at row 1, col 2
    """
    if rng is None:
        rng = np.random.default_rng()

    assert grid.ndim == 2, "Grid must be 2D"
    assert 0 <= pad_value <= 9, "Pad value must be 0-9"

    h, w = grid.shape
    target_h, target_w = target_size

    assert h <= target_h, f"Grid height {h} > target height {target_h}"
    assert w <= target_w, f"Grid width {w} > target width {target_w}"

    # Determine position
    if random_position:
        max_row_offset = target_h - h
        max_col_offset = target_w - w
        row_offset = rng.integers(0, max_row_offset + 1)
        col_offset = rng.integers(0, max_col_offset + 1)
    else:
        row_offset = 0
        col_offset = 0

    # Create padded grid
    padded = np.full(target_size, pad_value, dtype=grid.dtype)
    padded[row_offset:row_offset + h, col_offset:col_offset + w] = grid

    return padded, (row_offset, col_offset)


def dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply dihedral group transformation (D4 symmetry group).

    The dihedral group D4 has 8 elements:
    - Identity (0)
    - 3 rotations: 90°, 180°, 270° (1, 2, 3)
    - 4 reflections: horizontal, vertical, diagonal, anti-diagonal (4, 5, 6, 7)

    Args:
        grid: Input grid (H, W)
        transform_id: Transform ID (0-7)

    Returns:
        Transformed grid

    Example:
        >>> grid = np.array([[1, 2], [3, 4]])
        >>> for i in range(8):
        ...     transformed = dihedral_transform(grid, i)
        ...     print(f"Transform {i}:")
        ...     print(transformed)
    """
    assert 0 <= transform_id <= 7, f"Transform ID must be 0-7, got {transform_id}"

    if transform_id == 0:
        # Identity
        return grid.copy()
    elif transform_id == 1:
        # Rotate 90°
        return rotate_grid(grid, RotationType.ROTATE_90)
    elif transform_id == 2:
        # Rotate 180°
        return rotate_grid(grid, RotationType.ROTATE_180)
    elif transform_id == 3:
        # Rotate 270°
        return rotate_grid(grid, RotationType.ROTATE_270)
    elif transform_id == 4:
        # Reflect horizontal
        return reflect_grid(grid, ReflectionType.HORIZONTAL)
    elif transform_id == 5:
        # Reflect vertical
        return reflect_grid(grid, ReflectionType.VERTICAL)
    elif transform_id == 6:
        # Reflect diagonal
        return reflect_grid(grid, ReflectionType.DIAGONAL)
    elif transform_id == 7:
        # Reflect anti-diagonal
        return reflect_grid(grid, ReflectionType.ANTI_DIAGONAL)


def inverse_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """
    Apply inverse of dihedral transformation.

    Args:
        grid: Input grid (H, W)
        transform_id: Transform ID (0-7) to invert

    Returns:
        Inverse transformed grid
    """
    # Inverse mappings for dihedral group
    inverse_map = {
        0: 0,  # Identity is self-inverse
        1: 3,  # Rotate 90° inverse is rotate 270°
        2: 2,  # Rotate 180° is self-inverse
        3: 1,  # Rotate 270° inverse is rotate 90°
        4: 4,  # Reflections are self-inverse
        5: 5,
        6: 6,
        7: 7,
    }

    return dihedral_transform(grid, inverse_map[transform_id])


def augment_grid(
    grid: np.ndarray,
    config: AugmentationConfig,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, dict]:
    """
    Apply full augmentation pipeline to a grid.

    Args:
        grid: Input grid (H, W)
        config: Augmentation configuration
        rng: Random number generator

    Returns:
        Tuple of (augmented_grid, augmentation_info)
        augmentation_info contains details about transformations applied

    Example:
        >>> config = AugmentationConfig(
        ...     enable_color_shuffle=True,
        ...     enable_rotation=True,
        ...     target_size=(30, 30)
        ... )
        >>> augmented, info = augment_grid(grid, config)
        >>> print(info)
        {
            'color_mapping': array([0, 3, 1, ...]),
            'dihedral_id': 2,
            'translation_offset': (5, 8)
        }
    """
    if rng is None:
        if config.seed is not None:
            rng = np.random.default_rng(config.seed)
        else:
            rng = np.random.default_rng()

    augmented = grid.copy()
    info = {}

    # Step 1: Dihedral transformation (rotation/reflection)
    if config.enable_rotation or config.enable_reflection:
        dihedral_id = rng.integers(0, 8)
        augmented = dihedral_transform(augmented, dihedral_id)
        info['dihedral_id'] = int(dihedral_id)
    else:
        info['dihedral_id'] = 0

    # Step 2: Color shuffle
    if config.enable_color_shuffle:
        augmented, color_mapping = color_shuffle(
            augmented,
            keep_black=config.color_shuffle_keep_black,
            rng=rng
        )
        info['color_mapping'] = color_mapping
    else:
        info['color_mapping'] = np.arange(10, dtype=np.uint8)

    # Step 3: Translation (padding)
    if config.enable_translation and config.target_size is not None:
        augmented, offset = translate_grid(
            augmented,
            target_size=config.target_size,
            random_position=True,
            rng=rng
        )
        info['translation_offset'] = offset
    else:
        info['translation_offset'] = (0, 0)

    return augmented, info


def augment_example(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    config: AugmentationConfig,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply same augmentation to both input and output grids.

    This ensures the transformation pattern is preserved.

    Args:
        input_grid: Input grid (H, W)
        output_grid: Output grid (H', W')
        config: Augmentation configuration
        rng: Random number generator

    Returns:
        Tuple of (aug_input, aug_output, augmentation_info)

    Example:
        >>> config = AugmentationConfig(seed=42)
        >>> aug_input, aug_output, info = augment_example(
        ...     puzzle.train[0].input,
        ...     puzzle.train[0].output,
        ...     config
        ... )
    """
    if rng is None:
        if config.seed is not None:
            rng = np.random.default_rng(config.seed)
        else:
            rng = np.random.default_rng()

    # Generate augmentation parameters once
    info = {}

    # Dihedral transformation
    if config.enable_rotation or config.enable_reflection:
        dihedral_id = rng.integers(0, 8)
        info['dihedral_id'] = int(dihedral_id)
    else:
        dihedral_id = 0
        info['dihedral_id'] = 0

    # Color shuffle
    if config.enable_color_shuffle:
        if config.color_shuffle_keep_black:
            color_mapping = np.zeros(10, dtype=np.uint8)
            color_mapping[0] = 0
            color_mapping[1:] = rng.permutation(np.arange(1, 10, dtype=np.uint8))
        else:
            color_mapping = rng.permutation(np.arange(10, dtype=np.uint8))
        info['color_mapping'] = color_mapping
    else:
        color_mapping = np.arange(10, dtype=np.uint8)
        info['color_mapping'] = color_mapping

    # Apply to input
    aug_input = dihedral_transform(input_grid, dihedral_id)
    aug_input = color_mapping[aug_input]

    # Apply to output (same transformations!)
    aug_output = dihedral_transform(output_grid, dihedral_id)
    aug_output = color_mapping[aug_output]

    # Translation (only if target size specified)
    if config.enable_translation and config.target_size is not None:
        # For input and output, we might want different handling
        # Here we pad both, but could be customized
        aug_input, offset_input = translate_grid(
            aug_input,
            target_size=config.target_size,
            random_position=True,
            rng=rng
        )
        aug_output, offset_output = translate_grid(
            aug_output,
            target_size=config.target_size,
            random_position=True,
            rng=rng
        )
        info['translation_offset_input'] = offset_input
        info['translation_offset_output'] = offset_output
    else:
        info['translation_offset_input'] = (0, 0)
        info['translation_offset_output'] = (0, 0)

    return aug_input, aug_output, info
