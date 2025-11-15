#!/usr/bin/env python3
"""
Example 3: Full Dataset Processing Pipeline

Demonstrates how to process ARC-AGI-1 dataset with augmentation,
using BOTH train and test examples for training.

Key Point: We use ALL examples (train + test) from each puzzle during training!
This gives the model more examples to learn the transformation pattern from.
"""

import sys
from pathlib import Path
import numpy as np
from typing import List, Dict
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_dataset import (
    load_arc_dataset,
    ARCPuzzle,
    ARCExample,
    AugmentationConfig,
    augment_example,
    DatasetConfig,
)


def grid_hash(grid: np.ndarray) -> str:
    """Compute hash of grid for deduplication."""
    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def augment_puzzle(
    puzzle: ARCPuzzle,
    num_augmentations: int,
    config: AugmentationConfig,
) -> List[Dict]:
    """
    Augment a single puzzle.

    IMPORTANT: We use ALL examples (train + test) for training!

    Args:
        puzzle: ARC puzzle with train and test examples
        num_augmentations: Number of augmented versions to create
        config: Augmentation configuration

    Returns:
        List of augmented example dictionaries
    """
    # Combine ALL examples (train + test) for augmentation
    all_examples = puzzle.train + puzzle.test

    print(f"\nPuzzle {puzzle.puzzle_id}:")
    print(f"  Train examples: {puzzle.num_train}")
    print(f"  Test examples: {puzzle.num_test}")
    print(f"  Total examples for training: {len(all_examples)}")

    # Store all augmented examples
    augmented_data = []

    # Keep track of unique grids (for deduplication)
    seen_hashes = set()

    # Original examples (no augmentation)
    for ex_idx, example in enumerate(all_examples):
        input_hash = grid_hash(example.input)
        output_hash = grid_hash(example.output)
        pair_hash = input_hash + output_hash

        if pair_hash not in seen_hashes:
            augmented_data.append({
                'puzzle_id': puzzle.puzzle_id,
                'augmentation_id': 0,  # Original
                'example_idx': ex_idx,
                'input': example.input,
                'output': example.output,
                'is_original': True,
            })
            seen_hashes.add(pair_hash)

    # Generate augmentations
    rng = np.random.default_rng(config.seed)

    for aug_id in range(1, num_augmentations + 1):
        # Apply same augmentation to all examples in puzzle
        for ex_idx, example in enumerate(all_examples):
            # Create fresh RNG for this specific augmentation
            aug_rng = np.random.default_rng(config.seed + aug_id * 1000 + ex_idx)

            aug_input, aug_output, info = augment_example(
                example.input,
                example.output,
                config,
                rng=aug_rng
            )

            # Check for duplicates
            input_hash = grid_hash(aug_input)
            output_hash = grid_hash(aug_output)
            pair_hash = input_hash + output_hash

            if pair_hash not in seen_hashes:
                augmented_data.append({
                    'puzzle_id': puzzle.puzzle_id,
                    'augmentation_id': aug_id,
                    'example_idx': ex_idx,
                    'input': aug_input,
                    'output': aug_output,
                    'is_original': False,
                    'transform_info': info,
                })
                seen_hashes.add(pair_hash)

    unique_count = len(augmented_data)
    expected_count = len(all_examples) * (num_augmentations + 1)

    print(f"  Generated: {unique_count} unique examples (expected {expected_count})")
    print(f"  Duplicates removed: {expected_count - unique_count}")

    return augmented_data


def process_dataset(dataset_config: DatasetConfig):
    """
    Process full dataset with augmentation.

    Args:
        dataset_config: Configuration for dataset processing
    """
    print("=" * 60)
    print("Processing ARC-AGI-1 Dataset")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Subsets: {dataset_config.subsets}")
    print(f"  Augmentations per puzzle: {dataset_config.num_augmentations}")
    print(f"  Color shuffle: {dataset_config.enable_color_shuffle}")
    print(f"  Rotation/Reflection: {dataset_config.enable_rotation}")
    print(f"  Translation: {dataset_config.enable_translation}")
    print(f"  Target size: {dataset_config.target_grid_size}")

    all_processed_data = []

    for subset in dataset_config.subsets:
        print(f"\n{'=' * 60}")
        print(f"Processing subset: {subset}")
        print(f"{'=' * 60}")

        # Load dataset
        dataset = load_arc_dataset(
            dataset_name=subset,
            data_dir=dataset_config.data_dir,
            max_puzzles=dataset_config.max_puzzles
        )

        print(f"\nLoaded {len(dataset)} puzzles")

        # Create augmentation config
        aug_config = AugmentationConfig(
            enable_color_shuffle=dataset_config.enable_color_shuffle,
            enable_rotation=dataset_config.enable_rotation,
            enable_reflection=dataset_config.enable_reflection,
            enable_translation=dataset_config.enable_translation,
            target_size=dataset_config.target_grid_size,
            color_shuffle_keep_black=dataset_config.color_shuffle_keep_black,
            seed=dataset_config.seed,
        )

        # Process each puzzle
        subset_data = []
        for puzzle in dataset.puzzles:
            puzzle_data = augment_puzzle(
                puzzle,
                num_augmentations=dataset_config.num_augmentations,
                config=aug_config
            )
            subset_data.extend(puzzle_data)

        all_processed_data.extend(subset_data)

        print(f"\nSubset {subset} complete:")
        print(f"  Total examples: {len(subset_data)}")

    print(f"\n{'=' * 60}")
    print(f"Processing Complete!")
    print(f"{'=' * 60}")
    print(f"Total examples across all subsets: {len(all_processed_data)}")

    return all_processed_data


def demonstrate_small_dataset():
    """Demonstrate processing with a small dataset."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Small Dataset Processing")
    print("=" * 60)

    # Use debug config (5 puzzles, 10 augmentations)
    from arc_dataset.config import get_debug_config

    config = get_debug_config()
    processed_data = process_dataset(config)

    # Show sample of processed data
    print("\n" + "=" * 60)
    print("Sample Processed Examples")
    print("=" * 60)

    for i, example in enumerate(processed_data[:3]):
        print(f"\nExample {i + 1}:")
        print(f"  Puzzle ID: {example['puzzle_id']}")
        print(f"  Augmentation ID: {example['augmentation_id']}")
        print(f"  Original: {example['is_original']}")
        print(f"  Input shape: {example['input'].shape}")
        print(f"  Output shape: {example['output'].shape}")
        if not example['is_original']:
            info = example['transform_info']
            print(f"  Dihedral transform: {info['dihedral_id']}")

    # Statistics by puzzle
    print("\n" + "=" * 60)
    print("Examples per Puzzle")
    print("=" * 60)

    puzzle_counts = {}
    for example in processed_data:
        pid = example['puzzle_id']
        puzzle_counts[pid] = puzzle_counts.get(pid, 0) + 1

    for pid, count in list(puzzle_counts.items())[:5]:
        print(f"  {pid}: {count} examples")


def main():
    """Run the demonstration."""
    demonstrate_small_dataset()

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("""
1. We use BOTH train and test examples for training
   - This gives the model more examples per puzzle
   - Model learns from 6 examples instead of 5

2. Same puzzle_id for all examples
   - Original + augmented versions share the same ID
   - Model learns puzzle-specific patterns

3. Augmentation creates diverse training data
   - Rotation, reflection, color shuffle
   - Helps model learn transformation invariances

4. Deduplication prevents redundant examples
   - Hash-based checking ensures uniqueness
    """)


if __name__ == "__main__":
    main()
