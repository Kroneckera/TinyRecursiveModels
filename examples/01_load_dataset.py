#!/usr/bin/env python3
"""
Example 1: Load ARC-AGI-1 Dataset

Demonstrates how to load and explore the ARC-AGI-1 dataset.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_dataset import load_arc_dataset


def main():
    print("=" * 60)
    print("Loading ARC-AGI-1 Training Dataset")
    print("=" * 60)

    # Load training dataset
    dataset = load_arc_dataset(
        dataset_name="training",
        data_dir="kaggle/combined",
        max_puzzles=5  # Load only 5 puzzles for this example
    )

    print(f"\nLoaded {len(dataset)} puzzles")

    # Get statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Total puzzles: {stats['num_puzzles']}")
    print(f"  Total train examples: {stats['total_train_examples']}")
    print(f"  Total test examples: {stats['total_test_examples']}")
    print(f"  Avg train per puzzle: {stats['avg_train_per_puzzle']:.2f}")
    print(f"  Avg test per puzzle: {stats['avg_test_per_puzzle']:.2f}")

    # Explore first puzzle
    print("\n" + "=" * 60)
    print("Exploring First Puzzle")
    print("=" * 60)

    puzzle = dataset[0]
    print(f"\nPuzzle ID: {puzzle.puzzle_id}")
    print(f"Number of train examples: {puzzle.num_train}")
    print(f"Number of test examples: {puzzle.num_test}")

    # Show training examples
    print("\nTraining Examples:")
    for i, example in enumerate(puzzle.train):
        print(f"  Example {i + 1}:")
        print(f"    Input shape:  {example.input.shape}")
        print(f"    Output shape: {example.output.shape}")
        print(f"    Input:\n{example.input}")
        print(f"    Output:\n{example.output}")
        print()

    # Show test examples
    print("Test Examples:")
    for i, example in enumerate(puzzle.test):
        print(f"  Example {i + 1}:")
        print(f"    Input shape:  {example.input.shape}")
        print(f"    Output shape: {example.output.shape}")
        print(f"    Input:\n{example.input}")
        print()

    # Access puzzle by ID
    print("=" * 60)
    print("Access Puzzle by ID")
    print("=" * 60)

    puzzle_by_id = dataset.get_by_id(puzzle.puzzle_id)
    if puzzle_by_id:
        print(f"\nFound puzzle: {puzzle_by_id.puzzle_id}")
        print(f"Same as first puzzle: {puzzle_by_id is puzzle}")


if __name__ == "__main__":
    main()
