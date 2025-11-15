#!/usr/bin/env python3
"""
Example 2: Grid Augmentation

Demonstrates various augmentation operations on ARC grids.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_dataset import (
    load_arc_dataset,
    AugmentationConfig,
    augment_example,
    color_shuffle,
    rotate_grid,
    reflect_grid,
    dihedral_transform,
    RotationType,
    ReflectionType,
)


def print_grid(grid: np.ndarray, title: str = ""):
    """Pretty print a grid."""
    if title:
        print(f"\n{title}:")
    for row in grid:
        print("  " + " ".join(str(x) for x in row))


def demonstrate_color_shuffle():
    """Demonstrate color shuffling."""
    print("\n" + "=" * 60)
    print("Color Shuffling")
    print("=" * 60)

    grid = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 0, 1],
    ], dtype=np.uint8)

    print_grid(grid, "Original Grid")

    # Shuffle with black fixed
    rng = np.random.default_rng(42)
    shuffled, mapping = color_shuffle(grid, keep_black=True, rng=rng)

    print(f"\nColor Mapping: {mapping}")
    print_grid(shuffled, "Shuffled Grid (black fixed)")

    # Verify black (0) stayed the same
    print(f"\nBlack (0) positions unchanged: {np.all(grid == 0) == np.all(shuffled == 0)}")


def demonstrate_rotation():
    """Demonstrate rotation."""
    print("\n" + "=" * 60)
    print("Rotation")
    print("=" * 60)

    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.uint8)

    print_grid(grid, "Original Grid")
    print_grid(rotate_grid(grid, RotationType.ROTATE_90), "Rotate 90°")
    print_grid(rotate_grid(grid, RotationType.ROTATE_180), "Rotate 180°")
    print_grid(rotate_grid(grid, RotationType.ROTATE_270), "Rotate 270°")


def demonstrate_reflection():
    """Demonstrate reflection."""
    print("\n" + "=" * 60)
    print("Reflection")
    print("=" * 60)

    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.uint8)

    print_grid(grid, "Original Grid")
    print_grid(reflect_grid(grid, ReflectionType.HORIZONTAL), "Flip Horizontal")
    print_grid(reflect_grid(grid, ReflectionType.VERTICAL), "Flip Vertical")
    print_grid(reflect_grid(grid, ReflectionType.DIAGONAL), "Flip Diagonal")
    print_grid(reflect_grid(grid, ReflectionType.ANTI_DIAGONAL), "Flip Anti-Diagonal")


def demonstrate_dihedral():
    """Demonstrate all 8 dihedral transformations."""
    print("\n" + "=" * 60)
    print("Dihedral Group (D4) - All 8 Transformations")
    print("=" * 60)

    grid = np.array([
        [1, 2],
        [3, 4],
    ], dtype=np.uint8)

    print_grid(grid, "Original Grid")

    transform_names = [
        "Identity",
        "Rotate 90°",
        "Rotate 180°",
        "Rotate 270°",
        "Flip Horizontal",
        "Flip Vertical",
        "Flip Diagonal",
        "Flip Anti-Diagonal"
    ]

    for i in range(8):
        transformed = dihedral_transform(grid, i)
        print_grid(transformed, f"Transform {i}: {transform_names[i]}")


def demonstrate_full_augmentation():
    """Demonstrate full augmentation pipeline on real puzzle."""
    print("\n" + "=" * 60)
    print("Full Augmentation Pipeline on Real Puzzle")
    print("=" * 60)

    # Load a real puzzle
    dataset = load_arc_dataset("training", max_puzzles=1)
    puzzle = dataset[0]
    example = puzzle.train[0]

    print(f"\nPuzzle ID: {puzzle.puzzle_id}")
    print_grid(example.input, "Original Input")
    print_grid(example.output, "Original Output")

    # Apply augmentation
    config = AugmentationConfig(
        enable_color_shuffle=True,
        enable_rotation=True,
        enable_reflection=True,
        enable_translation=False,  # Disable padding for clearer visualization
        seed=42
    )

    aug_input, aug_output, info = augment_example(
        example.input,
        example.output,
        config
    )

    print(f"\nAugmentation Info:")
    print(f"  Dihedral transform: {info['dihedral_id']}")
    print(f"  Color mapping: {info['color_mapping']}")

    print_grid(aug_input, "Augmented Input")
    print_grid(aug_output, "Augmented Output")

    # Generate multiple augmentations
    print("\n" + "=" * 60)
    print("Multiple Random Augmentations")
    print("=" * 60)

    for i in range(3):
        config_i = AugmentationConfig(
            enable_color_shuffle=True,
            enable_rotation=True,
            enable_reflection=True,
            enable_translation=False,
            seed=42 + i  # Different seed
        )

        aug_input_i, aug_output_i, info_i = augment_example(
            example.input,
            example.output,
            config_i
        )

        print(f"\nAugmentation #{i + 1}:")
        print(f"  Dihedral: {info_i['dihedral_id']}, Colors: {list(info_i['color_mapping'][:5])}...")
        print_grid(aug_input_i, f"  Input")


def main():
    """Run all demonstrations."""
    demonstrate_color_shuffle()
    demonstrate_rotation()
    demonstrate_reflection()
    demonstrate_dihedral()
    demonstrate_full_augmentation()

    print("\n" + "=" * 60)
    print("All augmentation demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
