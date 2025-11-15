### ARC-AGI-1 Dataset Foundation

A clean, well-documented codebase for loading and augmenting ARC-AGI-1 puzzles.

## Overview

This package provides:
- **Clean dataset loading** for ARC-AGI-1 challenges
- **Configurable augmentation** (color shuffle, rotation, reflection, translation)
- **Simple API** for processing puzzles
- **Full documentation** and examples

## Key Design Principles

### 1. Use ALL Examples for Training

**Important**: We use BOTH train and test examples during training!

```python
# For each puzzle:
puzzle.train = [ex1, ex2, ex3, ex4, ex5]  # 5 train examples
puzzle.test  = [ex_test]                  # 1 test example

# During training, we use ALL 6 examples:
training_examples = puzzle.train + puzzle.test  # 6 total examples
```

**Why?** This gives the model more examples to learn the transformation pattern from each puzzle.

### 2. Puzzle-Specific Learning

All examples from the same puzzle (original + augmented) share the **same puzzle_id**:

```python
puzzle_id = 123
examples = [
    (input1, output1, puzzle_id=123),  # Original train ex 1
    (input2, output2, puzzle_id=123),  # Original train ex 2
    ...
    (input_test, output_test, puzzle_id=123),  # Original test ex
    (aug_input1, aug_output1, puzzle_id=123),  # Augmented version
    ...
]
```

The model learns: "puzzle_id=123 → transformation pattern T"

### 3. Semantic-Preserving Augmentation

Augmentations preserve the puzzle's transformation rule:

- **Color shuffle**: Permute colors (keep black/0 fixed)
- **Rotation**: 0°, 90°, 180°, 270°
- **Reflection**: Horizontal, vertical, diagonal, anti-diagonal
- **Translation**: Random padding within 30×30 grid

**Critical**: Same augmentation applied to BOTH input and output!

## Installation

No installation needed - just Python 3.8+ and NumPy:

```bash
pip install numpy
```

## Quick Start

### Load Dataset

```python
from arc_dataset import load_arc_dataset

# Load training data
dataset = load_arc_dataset('training')

print(f"Loaded {len(dataset)} puzzles")

# Get first puzzle
puzzle = dataset[0]
print(f"Puzzle {puzzle.puzzle_id}:")
print(f"  Train examples: {puzzle.num_train}")
print(f"  Test examples: {puzzle.num_test}")

# Access examples
for example in puzzle.train:
    print(f"  Input: {example.input.shape}, Output: {example.output.shape}")
```

### Augment Examples

```python
from arc_dataset import AugmentationConfig, augment_example

# Configure augmentation
config = AugmentationConfig(
    enable_color_shuffle=True,
    enable_rotation=True,
    enable_reflection=True,
    enable_translation=False,
    seed=42
)

# Augment an example
aug_input, aug_output, info = augment_example(
    example.input,
    example.output,
    config
)

print(f"Augmentation info: {info}")
```

### Full Pipeline

```python
from arc_dataset import DatasetConfig
from arc_dataset.config import get_debug_config

# Get configuration
config = get_debug_config()  # Small dataset for testing

# Or create custom config
config = DatasetConfig(
    data_dir="kaggle/combined",
    subsets=["training"],
    num_augmentations=1000,
    target_grid_size=(30, 30),
    seed=42
)

# Process dataset
# See examples/03_full_pipeline.py for complete implementation
```

## API Reference

### Core Classes

#### `ARCExample`
```python
@dataclass
class ARCExample:
    input: np.ndarray   # Shape: (H, W), values: 0-9
    output: np.ndarray  # Shape: (H', W'), values: 0-9
```

#### `ARCPuzzle`
```python
@dataclass
class ARCPuzzle:
    puzzle_id: str
    train: List[ARCExample]
    test: List[ARCExample]

    @property
    def num_train(self) -> int
    @property
    def num_test(self) -> int
```

#### `ARCDataset`
```python
class ARCDataset:
    def __init__(
        self,
        challenges_path: str,
        solutions_path: Optional[str] = None,
        max_puzzles: Optional[int] = None
    )

    def __len__(self) -> int
    def __getitem__(self, idx: int) -> ARCPuzzle
    def get_by_id(self, puzzle_id: str) -> Optional[ARCPuzzle]
    def get_statistics(self) -> Dict
```

### Augmentation Functions

#### `color_shuffle`
```python
def color_shuffle(
    grid: np.ndarray,
    keep_black: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle colors in grid."""
```

#### `rotate_grid`
```python
def rotate_grid(
    grid: np.ndarray,
    rotation: RotationType  # ROTATE_90, ROTATE_180, ROTATE_270
) -> np.ndarray:
    """Rotate grid by 90°, 180°, or 270° clockwise."""
```

#### `reflect_grid`
```python
def reflect_grid(
    grid: np.ndarray,
    reflection: ReflectionType  # HORIZONTAL, VERTICAL, DIAGONAL, ANTI_DIAGONAL
) -> np.ndarray:
    """Reflect grid along specified axis."""
```

#### `dihedral_transform`
```python
def dihedral_transform(
    grid: np.ndarray,
    transform_id: int  # 0-7
) -> np.ndarray:
    """Apply one of 8 dihedral group transformations."""
```

#### `augment_example`
```python
def augment_example(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    config: AugmentationConfig,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Apply same augmentation to both input and output."""
```

### Configuration

#### `AugmentationConfig`
```python
@dataclass
class AugmentationConfig:
    enable_color_shuffle: bool = True
    enable_rotation: bool = True
    enable_reflection: bool = True
    enable_translation: bool = True
    target_size: Optional[Tuple[int, int]] = (30, 30)
    color_shuffle_keep_black: bool = True
    seed: Optional[int] = None
```

#### `DatasetConfig`
```python
@dataclass
class DatasetConfig:
    data_dir: str = "kaggle/combined"
    subsets: List[str] = ["training"]
    num_augmentations: int = 1000
    enable_color_shuffle: bool = True
    enable_rotation: bool = True
    enable_reflection: bool = True
    enable_translation: bool = True
    target_grid_size: Optional[Tuple[int, int]] = (30, 30)
    seed: int = 42
    max_puzzles: Optional[int] = None
    # ... more options
```

## Examples

See `examples/` directory:

1. **`01_load_dataset.py`**: Load and explore ARC-AGI-1 dataset
2. **`02_augmentation.py`**: Demonstrate all augmentation operations
3. **`03_full_pipeline.py`**: Complete processing pipeline

Run examples:
```bash
cd examples
python 01_load_dataset.py
python 02_augmentation.py
python 03_full_pipeline.py
```

## Data Format

### Input: ARC-AGI JSON Format

```json
{
  "007bbfb7": {
    "train": [
      {"input": [[0,7,7], ...], "output": [[0,0,0], ...]},
      {"input": [[...]], "output": [[...]]},
      ...
    ],
    "test": [
      {"input": [[7,0,7], ...]}
    ]
  }
}
```

### Output: Augmented Examples

Each example is a dictionary:
```python
{
    'puzzle_id': '007bbfb7',
    'augmentation_id': 42,  # 0 = original, 1+ = augmented
    'example_idx': 0,       # Which example from puzzle
    'input': np.array(...), # Shape: (H, W) or (30, 30) if padded
    'output': np.array(...),
    'is_original': False,
    'transform_info': {
        'dihedral_id': 2,
        'color_mapping': array([0, 3, 1, ...]),
        'translation_offset_input': (5, 8),
        'translation_offset_output': (3, 2)
    }
}
```

## Design Decisions

### Why Use Both Train and Test Examples?

In ARC-AGI, each puzzle has:
- 3-5 training examples (demonstrating the pattern)
- 1 test example (apply the pattern)

Traditional approach: Train on train examples, evaluate on test examples.

**Our approach**: Use ALL examples (train + test) during model training.

**Rationale**:
1. More examples per puzzle → better pattern learning
2. We're not doing in-context learning, so no "contamination"
3. Model learns via puzzle embeddings, not example memorization
4. At inference, model applies learned pattern to truly new puzzles

### Why Same Puzzle ID for All Examples?

The model architecture uses **puzzle-specific embeddings**:

```python
# Simplified model forward pass
puzzle_embedding = embedding_table[puzzle_id]
output = model(input, puzzle_embedding)
```

By giving all examples (and augmentations) the same `puzzle_id`, the model learns:
```
"puzzle_id=123" → "transformation pattern T"
```

This is stored in the learned embedding for puzzle 123.

### Why Dihedral Group (8 Transformations)?

The dihedral group D4 consists of all symmetries of a square:
- 4 rotations (0°, 90°, 180°, 270°)
- 4 reflections (H, V, diagonal, anti-diagonal)

These are **semantically meaningful** for grid transformations and preserve the puzzle's logic.

## File Structure

```
arc_dataset/
├── __init__.py          # Package exports
├── loader.py            # Dataset loading
├── augmentation.py      # Augmentation functions
├── config.py            # Configuration classes
└── README.md            # This file

examples/
├── 01_load_dataset.py   # Basic loading
├── 02_augmentation.py   # Augmentation demo
└── 03_full_pipeline.py  # Complete pipeline

kaggle/combined/         # ARC-AGI data files
├── arc-agi_training_challenges.json
├── arc-agi_training_solutions.json
├── arc-agi_evaluation_challenges.json
└── ...
```

## Common Patterns

### Load and Augment Single Puzzle

```python
from arc_dataset import load_arc_dataset, AugmentationConfig, augment_example

# Load
dataset = load_arc_dataset('training', max_puzzles=1)
puzzle = dataset[0]

# Combine all examples
all_examples = puzzle.train + puzzle.test

# Augment
config = AugmentationConfig(seed=42)

for example in all_examples:
    for i in range(10):  # 10 augmentations
        aug_input, aug_output, info = augment_example(
            example.input,
            example.output,
            config
        )
        # Process augmented example...
```

### Batch Processing

```python
from arc_dataset import load_arc_dataset, DatasetConfig

config = DatasetConfig(
    subsets=["training"],
    num_augmentations=1000,
    max_puzzles=None  # Process all
)

dataset = load_arc_dataset('training')

all_augmented_examples = []

for puzzle in dataset:
    all_examples = puzzle.train + puzzle.test

    for example in all_examples:
        # Original
        all_augmented_examples.append({
            'puzzle_id': puzzle.puzzle_id,
            'input': example.input,
            'output': example.output,
        })

        # Augmented
        for aug_id in range(config.num_augmentations):
            aug_input, aug_output, info = augment_example(...)
            all_augmented_examples.append({
                'puzzle_id': puzzle.puzzle_id,
                'input': aug_input,
                'output': aug_output,
            })

print(f"Total examples: {len(all_augmented_examples)}")
```

## Testing

Run example scripts to verify installation:

```bash
# Test loading
python examples/01_load_dataset.py

# Test augmentation
python examples/02_augmentation.py

# Test full pipeline
python examples/03_full_pipeline.py
```

Expected output: No errors, displays puzzle information and augmented grids.

## Troubleshooting

**Q: FileNotFoundError for JSON files**
```
A: Ensure kaggle/combined/ directory exists with ARC-AGI JSON files.
   Download from: https://github.com/fchollet/ARC-AGI
```

**Q: All augmentations look the same**
```
A: Check that you're using different seeds for each augmentation.
   Or set seed=None for true randomness.
```

**Q: Import errors**
```
A: Add parent directory to Python path:
   sys.path.insert(0, str(Path(__file__).parent.parent))
```

## Next Steps

This foundation code provides:
- ✅ Clean dataset loading
- ✅ Flexible augmentation
- ✅ Configuration system
- ✅ Example scripts

To build a training pipeline, you'll need to add:
- [ ] Sequence conversion (grid → tokens)
- [ ] Batch sampling
- [ ] Model architecture
- [ ] Training loop
- [ ] Evaluation

See `TRAINING_PIPELINE.md` for details on the complete training process.

## License

This code is based on the TinyRecursiveModels repository.
See main repository README for license information.
