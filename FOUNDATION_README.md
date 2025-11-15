# ARC-AGI-1 Foundation Code

Clean, well-documented foundation codebase for ARC-AGI-1 dataset handling and augmentation.

## 🎯 What This Provides

A minimal, production-ready implementation of:

### ✅ Dataset Loading
- Load ARC-AGI-1 challenges (training, evaluation, concept)
- Clean dataclass-based API
- Automatic validation of grids
- Support for both challenges and solutions

### ✅ Data Augmentation
- **Color shuffle**: Permute colors 1-9 (keep black/0 fixed)
- **Rotation**: 90°, 180°, 270° clockwise
- **Reflection**: Horizontal, vertical, diagonal, anti-diagonal
- **Translation**: Random padding within target grid size
- **Dihedral group (D4)**: All 8 symmetry transformations
- **Composable**: Apply same transformations to input and output

### ✅ Configuration System
- Centralized configuration via dataclasses
- Preset configurations (default, debug, no-augmentation)
- Full control over augmentation parameters

### ✅ Examples & Documentation
- 3 complete example scripts
- Comprehensive API documentation
- Usage patterns and best practices

## 📁 Structure

```
arc_dataset/              # Main package
├── __init__.py          # Package exports
├── loader.py            # Dataset loading (ARCDataset, ARCPuzzle, ARCExample)
├── augmentation.py      # All augmentation functions
├── config.py            # Configuration classes
└── README.md            # Detailed API documentation

examples/                 # Example scripts
├── 01_load_dataset.py   # How to load ARC-AGI-1 data
├── 02_augmentation.py   # Augmentation demonstrations
└── 03_full_pipeline.py  # Complete processing pipeline

FOUNDATION_README.md      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install numpy
```

### 2. Load Dataset

```python
from arc_dataset import load_arc_dataset

# Load training data
dataset = load_arc_dataset('training')
print(f"Loaded {len(dataset)} puzzles")

# Access puzzle
puzzle = dataset[0]
print(f"Puzzle {puzzle.puzzle_id}: {puzzle.num_train} train, {puzzle.num_test} test")

# Access examples
example = puzzle.train[0]
print(f"Input shape: {example.input.shape}")
print(f"Output shape: {example.output.shape}")
```

### 3. Augment Data

```python
from arc_dataset import AugmentationConfig, augment_example

# Configure augmentation
config = AugmentationConfig(
    enable_color_shuffle=True,
    enable_rotation=True,
    enable_reflection=True,
    enable_translation=False,  # No padding
    seed=42
)

# Augment example (applies same transform to input and output)
aug_input, aug_output, info = augment_example(
    example.input,
    example.output,
    config
)

print(f"Original input: {example.input.shape}")
print(f"Augmented input: {aug_input.shape}")
print(f"Transform applied: dihedral_id={info['dihedral_id']}")
```

### 4. Process Full Dataset

```python
from arc_dataset import DatasetConfig

# Configure processing
config = DatasetConfig(
    subsets=["training"],
    num_augmentations=1000,
    target_grid_size=(30, 30),
    seed=42
)

# Process dataset (see examples/03_full_pipeline.py for complete code)
```

## 🔑 Key Design Decisions

### 1. Use ALL Examples for Training

**We use BOTH train and test examples during model training!**

```python
# Traditional approach
training_data = puzzle.train  # Only 5 examples

# Our approach
training_data = puzzle.train + puzzle.test  # 6 examples (5 train + 1 test)
```

**Why?**
- More examples per puzzle → better pattern learning
- Not doing in-context learning, so no "test contamination"
- Model learns via puzzle-specific embeddings
- At inference, applies learned pattern to truly NEW puzzles

### 2. Puzzle-Specific Learning

All examples from the same puzzle share the same `puzzle_id`:

```python
# All these have puzzle_id = "007bbfb7"
original_train_1 = (input1, output1, puzzle_id="007bbfb7")
original_train_2 = (input2, output2, puzzle_id="007bbfb7")
original_test = (input_test, output_test, puzzle_id="007bbfb7")
augmented_1 = (aug_input1, aug_output1, puzzle_id="007bbfb7")
augmented_2 = (aug_input2, aug_output2, puzzle_id="007bbfb7")
```

Model learns: "puzzle_id='007bbfb7' → specific transformation pattern T"

### 3. Semantic-Preserving Augmentation

Same augmentation applied to BOTH input and output:

```python
# ✅ CORRECT: Same transform
dihedral_id = random(0, 8)
color_map = random_permutation([1,2,3,4,5,6,7,8,9])

aug_input = apply_transforms(input, dihedral_id, color_map)
aug_output = apply_transforms(output, dihedral_id, color_map)  # Same transforms!

# ❌ WRONG: Different transforms
aug_input = apply_transforms(input, random_transform())
aug_output = apply_transforms(output, different_random_transform())  # Breaks pattern!
```

## 📊 Example Dataset Statistics

After augmentation (1000 augmentations per puzzle):

```
ARC-AGI-1 Training:
  Puzzles: 400
  Original examples: ~2,400 (400 puzzles × 6 examples avg)
  After augmentation: ~2,400,000 (2.4M examples)

ARC-AGI-1 Evaluation:
  Puzzles: 400
  Original examples: ~2,400
  After augmentation: ~2,400,000
```

## 🎓 API Overview

### Core Classes

```python
# Single example
@dataclass
class ARCExample:
    input: np.ndarray   # (H, W), values 0-9
    output: np.ndarray  # (H', W'), values 0-9

# Complete puzzle
@dataclass
class ARCPuzzle:
    puzzle_id: str
    train: List[ARCExample]  # 3-5 examples
    test: List[ARCExample]   # 1+ examples

# Dataset
class ARCDataset:
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> ARCPuzzle
    def get_by_id(self, puzzle_id: str) -> ARCPuzzle
```

### Augmentation Functions

```python
# Individual operations
color_shuffle(grid, keep_black=True) -> (aug_grid, mapping)
rotate_grid(grid, RotationType.ROTATE_90) -> aug_grid
reflect_grid(grid, ReflectionType.HORIZONTAL) -> aug_grid
dihedral_transform(grid, transform_id=0-7) -> aug_grid

# Full pipeline
augment_example(input, output, config) -> (aug_input, aug_output, info)
```

### Configuration

```python
# Augmentation config
@dataclass
class AugmentationConfig:
    enable_color_shuffle: bool = True
    enable_rotation: bool = True
    enable_reflection: bool = True
    enable_translation: bool = True
    target_size: Tuple[int, int] = (30, 30)
    seed: int = None

# Dataset config
@dataclass
class DatasetConfig:
    data_dir: str = "kaggle/combined"
    subsets: List[str] = ["training"]
    num_augmentations: int = 1000
    # ... + augmentation settings
```

## 📚 Examples

### Example 1: Load and Explore

```bash
python examples/01_load_dataset.py
```

Shows:
- How to load datasets
- Dataset statistics
- Puzzle structure
- Example grids

### Example 2: Augmentation

```bash
python examples/02_augmentation.py
```

Demonstrates:
- Color shuffling
- Rotation (90°, 180°, 270°)
- Reflection (4 types)
- Dihedral transformations (8 total)
- Full augmentation pipeline

### Example 3: Full Pipeline

```bash
python examples/03_full_pipeline.py
```

Complete implementation:
- Load dataset
- Apply augmentations
- Combine train + test examples
- Deduplicate
- Process multiple subsets

## 🔧 Configuration Presets

```python
from arc_dataset.config import (
    get_default_config,      # Full processing (1000 augs)
    get_debug_config,         # Quick testing (5 puzzles, 10 augs)
    get_no_augmentation_config,  # Original data only
)

# Use preset
config = get_debug_config()

# Or customize
config = DatasetConfig(
    subsets=["training", "evaluation"],
    num_augmentations=500,  # Fewer augmentations
    enable_translation=False,  # No padding
    max_puzzles=100,  # First 100 puzzles only
)
```

## 🧪 Testing

```bash
# Test loading
python examples/01_load_dataset.py

# Test augmentation
python examples/02_augmentation.py

# Test full pipeline
python examples/03_full_pipeline.py
```

Expected: Scripts run without errors, display puzzle information and visualizations.

## 🎯 What's Next?

This foundation provides clean data loading and augmentation. To build a complete training pipeline, you'll need:

1. **Sequence Conversion**: Grid → token sequences
   - Pad to 30×30
   - Flatten to 900 tokens
   - Add EOS markers

2. **Batch Sampling**: Efficient batch construction
   - Group-based sampling
   - Puzzle identifier tracking

3. **Model Architecture**: Neural network
   - Token embeddings
   - Puzzle embeddings
   - Recursive reasoning

4. **Training Loop**: Optimization
   - Loss computation
   - Dual optimizer (embeddings + weights)
   - Evaluation

See `TRAINING_PIPELINE.md` for the complete training process.

## 🐛 Troubleshooting

**Q: ModuleNotFoundError: No module named 'numpy'**
```bash
pip install numpy
```

**Q: FileNotFoundError for JSON files**
```
Ensure kaggle/combined/ directory exists with ARC-AGI files.
Download from: https://github.com/fchollet/ARC-AGI
```

**Q: All augmentations look identical**
```
Use different seeds for each augmentation, or set seed=None for randomness.
```

## 📝 Code Quality

This codebase follows best practices:

- ✅ **Type hints**: Full type annotations
- ✅ **Docstrings**: Comprehensive documentation
- ✅ **Validation**: Input checking and assertions
- ✅ **Clean API**: Intuitive, Pythonic interface
- ✅ **Examples**: Working code for all features
- ✅ **Modularity**: Composable functions
- ✅ **Configuration**: Centralized settings

## 📖 Additional Documentation

- `arc_dataset/README.md` - Complete API reference
- `examples/*.py` - Working examples with comments
- `TRAINING_PIPELINE.md` - Full training process
- `MODEL_IO_SHAPE_SUMMARY.md` - Model architecture details

## 🎓 Learning Path

1. **Start here**: `examples/01_load_dataset.py`
2. **Understand augmentation**: `examples/02_augmentation.py`
3. **See full pipeline**: `examples/03_full_pipeline.py`
4. **Read API docs**: `arc_dataset/README.md`
5. **Explore training**: `TRAINING_PIPELINE.md`

## 💡 Key Takeaways

1. **Use ALL examples**: train + test for better learning
2. **Same puzzle_id**: For all examples and augmentations
3. **Preserve semantics**: Same transform on input and output
4. **Augmentation is critical**: 1000× helps prevent overfitting
5. **Clean code matters**: This foundation makes training easier

## 🤝 Contributing

This is a clean foundation for ARC-AGI-1. Feel free to:
- Add new augmentation types
- Optimize performance
- Add preprocessing steps
- Build on top of this foundation

## 📄 License

See main repository for license information.

---

**Ready to use!** Start with `examples/01_load_dataset.py` and build from there.
