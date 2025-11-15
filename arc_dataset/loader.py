"""
ARC-AGI-1 Dataset Loader

Loads puzzles from the ARC-AGI-1 challenge format (JSON files).
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ARCExample:
    """A single input-output example from an ARC puzzle."""
    input: np.ndarray  # Shape: (H, W), values: 0-9
    output: np.ndarray  # Shape: (H', W'), values: 0-9

    def __post_init__(self):
        """Validate the example after initialization."""
        assert self.input.ndim == 2, f"Input must be 2D, got shape {self.input.shape}"
        assert self.output.ndim == 2, f"Output must be 2D, got shape {self.output.shape}"
        assert self.input.dtype == np.uint8, f"Input must be uint8, got {self.input.dtype}"
        assert self.output.dtype == np.uint8, f"Output must be uint8, got {self.output.dtype}"
        assert np.all((self.input >= 0) & (self.input <= 9)), "Input values must be 0-9"
        assert np.all((self.output >= 0) & (self.output <= 9)), "Output values must be 0-9"


@dataclass
class ARCPuzzle:
    """A complete ARC puzzle with train and test examples."""
    puzzle_id: str
    train: List[ARCExample]
    test: List[ARCExample]  # Test examples may have dummy outputs if solutions not available

    def __len__(self):
        """Total number of examples (train + test)."""
        return len(self.train) + len(self.test)

    @property
    def num_train(self) -> int:
        """Number of training examples."""
        return len(self.train)

    @property
    def num_test(self) -> int:
        """Number of test examples."""
        return len(self.test)


class ARCDataset:
    """
    ARC-AGI-1 Dataset Loader

    Loads puzzles from JSON files in the ARC challenge format.

    Example:
        >>> dataset = ARCDataset(
        ...     challenges_path="arc-agi_training_challenges.json",
        ...     solutions_path="arc-agi_training_solutions.json"
        ... )
        >>> print(f"Loaded {len(dataset)} puzzles")
        >>> puzzle = dataset[0]
        >>> print(f"Puzzle {puzzle.puzzle_id}: {puzzle.num_train} train, {puzzle.num_test} test")
    """

    def __init__(
        self,
        challenges_path: str,
        solutions_path: Optional[str] = None,
        max_puzzles: Optional[int] = None,
    ):
        """
        Initialize ARC dataset loader.

        Args:
            challenges_path: Path to challenges JSON file (contains inputs and train outputs)
            solutions_path: Path to solutions JSON file (contains test outputs).
                          If None, test examples will have dummy outputs.
            max_puzzles: Maximum number of puzzles to load (for debugging)
        """
        self.challenges_path = Path(challenges_path)
        self.solutions_path = Path(solutions_path) if solutions_path else None

        # Load puzzles
        self.puzzles = self._load_puzzles(max_puzzles)
        self.puzzle_ids = [p.puzzle_id for p in self.puzzles]

    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _parse_grid(self, grid_data: List[List[int]]) -> np.ndarray:
        """
        Parse grid from JSON format to numpy array.

        Args:
            grid_data: List of lists representing the grid

        Returns:
            np.ndarray of shape (H, W) with dtype uint8
        """
        grid = np.array(grid_data, dtype=np.uint8)

        # Validate
        assert grid.ndim == 2, f"Grid must be 2D, got shape {grid.shape}"
        assert grid.shape[0] > 0 and grid.shape[1] > 0, "Grid cannot be empty"
        assert grid.shape[0] <= 30 and grid.shape[1] <= 30, f"Grid too large: {grid.shape}"
        assert np.all((grid >= 0) & (grid <= 9)), "Grid values must be 0-9"

        return grid

    def _load_puzzles(self, max_puzzles: Optional[int] = None) -> List[ARCPuzzle]:
        """Load all puzzles from JSON files."""
        # Load challenges
        challenges_data = self._load_json(self.challenges_path)

        # Load solutions if available
        solutions_data = None
        if self.solutions_path and self.solutions_path.exists():
            solutions_data = self._load_json(self.solutions_path)

        puzzles = []
        for idx, (puzzle_id, puzzle_data) in enumerate(challenges_data.items()):
            if max_puzzles and idx >= max_puzzles:
                break

            # Parse training examples
            train_examples = []
            for ex in puzzle_data.get('train', []):
                train_examples.append(ARCExample(
                    input=self._parse_grid(ex['input']),
                    output=self._parse_grid(ex['output'])
                ))

            # Parse test examples
            test_examples = []
            for test_idx, ex in enumerate(puzzle_data.get('test', [])):
                input_grid = self._parse_grid(ex['input'])

                # Get output from solutions if available
                if solutions_data and puzzle_id in solutions_data:
                    output_grid = self._parse_grid(solutions_data[puzzle_id][test_idx])
                elif 'output' in ex:
                    # Some datasets include test outputs in challenges file
                    output_grid = self._parse_grid(ex['output'])
                else:
                    # Use dummy output (1×1 grid with value 0)
                    output_grid = np.array([[0]], dtype=np.uint8)

                test_examples.append(ARCExample(
                    input=input_grid,
                    output=output_grid
                ))

            puzzles.append(ARCPuzzle(
                puzzle_id=puzzle_id,
                train=train_examples,
                test=test_examples
            ))

        return puzzles

    def __len__(self) -> int:
        """Number of puzzles in the dataset."""
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> ARCPuzzle:
        """Get puzzle by index."""
        return self.puzzles[idx]

    def get_by_id(self, puzzle_id: str) -> Optional[ARCPuzzle]:
        """Get puzzle by ID."""
        for puzzle in self.puzzles:
            if puzzle.puzzle_id == puzzle_id:
                return puzzle
        return None

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        total_train = sum(p.num_train for p in self.puzzles)
        total_test = sum(p.num_test for p in self.puzzles)

        train_sizes = [(ex.input.shape, ex.output.shape)
                      for p in self.puzzles for ex in p.train]
        test_sizes = [(ex.input.shape, ex.output.shape)
                     for p in self.puzzles for ex in p.test]

        return {
            'num_puzzles': len(self.puzzles),
            'total_train_examples': total_train,
            'total_test_examples': total_test,
            'avg_train_per_puzzle': total_train / len(self.puzzles) if self.puzzles else 0,
            'avg_test_per_puzzle': total_test / len(self.puzzles) if self.puzzles else 0,
            'train_grid_sizes': train_sizes,
            'test_grid_sizes': test_sizes,
        }


def load_arc_dataset(
    dataset_name: str = "training",
    data_dir: str = "kaggle/combined",
    max_puzzles: Optional[int] = None,
) -> ARCDataset:
    """
    Convenience function to load ARC-AGI-1 datasets.

    Args:
        dataset_name: One of 'training', 'evaluation', or 'concept'
        data_dir: Directory containing the JSON files
        max_puzzles: Maximum number of puzzles to load

    Returns:
        ARCDataset instance

    Example:
        >>> train_data = load_arc_dataset('training')
        >>> eval_data = load_arc_dataset('evaluation')
    """
    data_dir = Path(data_dir)

    valid_names = ['training', 'evaluation', 'concept']
    if dataset_name not in valid_names:
        raise ValueError(f"dataset_name must be one of {valid_names}, got '{dataset_name}'")

    challenges_path = data_dir / f"arc-agi_{dataset_name}_challenges.json"
    solutions_path = data_dir / f"arc-agi_{dataset_name}_solutions.json"

    if not challenges_path.exists():
        raise FileNotFoundError(f"Challenges file not found: {challenges_path}")

    # Solutions may not exist for some datasets
    if not solutions_path.exists():
        print(f"Warning: Solutions file not found: {solutions_path}")
        solutions_path = None

    return ARCDataset(
        challenges_path=str(challenges_path),
        solutions_path=str(solutions_path) if solutions_path else None,
        max_puzzles=max_puzzles,
    )
