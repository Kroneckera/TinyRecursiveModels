# TRM Model Input/Output Shape Summary

## Overview
This document explains the input/output shapes of the Tiny Recursive Model (TRM) and how it processes test cases consisting of multiple input-output example pairs.

---

## What is ARC-AGI?

### The Challenge
**ARC-AGI (Abstraction and Reasoning Corpus)** is a benchmark designed by François Chollet to measure AI's ability to acquire new skills and solve novel problems through abstract reasoning, rather than pattern memorization.

**Key Properties:**
- Each puzzle demonstrates a unique transformation pattern
- Only 3-5 training examples provided per puzzle
- Tests genuine reasoning ability, not memorization
- Human-solvable (most humans can solve ~80%)
- Extremely challenging for AI systems (most LLMs <5%)

### Dataset Versions

This codebase supports multiple ARC-AGI dataset versions:

| Version | Subsets | Total Puzzles | Purpose |
|---------|---------|---------------|---------|
| **ARC-AGI-1** | training: 400<br>evaluation: 400<br>concept: 160 | 960 | Original 2019 benchmark |
| **ARC-AGI-2** | training2: 1000<br>evaluation2: 120<br>concept: 160 | 1280 | Expanded 2024 benchmark |

**Important**: Cannot train on both ARC-1 and ARC-2 together because ARC-2 training set contains some ARC-1 evaluation puzzles.

### TRM Performance

**State-of-the-art results with only 7M parameters:**
- **ARC-AGI-1**: 45% accuracy on evaluation set
- **ARC-AGI-2**: 8% accuracy on evaluation set

For context:
- Human performance: ~80%
- Previous SOTA (HRM): ~40% on ARC-1
- Most large language models: <5%

This demonstrates that **"less is more"** - recursive reasoning with a tiny network can achieve what billion-parameter models cannot.

---

## Data Format: ARC-AGI Puzzles

### Puzzle Structure
Each puzzle consists of:
- **Training examples**: Multiple input-output grid pairs (typically 2-5 pairs)
- **Test examples**: Input grids where the model must predict the output

### Example Structure
```json
{
  "puzzle_id": {
    "train": [
      {"input": [[0,7,7], [7,7,7], ...], "output": [[0,0,0], [7,7,7], ...]},
      {"input": [[...]], "output": [[...]]},
      ...
    ],
    "test": [
      {"input": [[7,0,7], [7,0,7], ...]}
    ]
  }
}
```

### Grid Properties
- **Grid values**: Integers 0-9 (representing colors)
- **Grid size**: Variable, up to 30×30 maximum
- **Input and output grids**: Can have different dimensions

---

## Data Preprocessing Pipeline

### 1. Grid to Sequence Conversion
Located in: `dataset/build_arc_dataset.py:np_grid_to_seq_translational_augment()`

#### Transformation Process:
1. **Value Mapping**:
   - PAD: `0`
   - EOS (end-of-sequence): `1`
   - Digit 0: `2`
   - Digit 1: `3`
   - ...
   - Digit 9: `11`

2. **Padding**:
   - All grids padded to 30×30 (ARCMaxGridSize)
   - Random translation augmentation applied during training (except one example per puzzle)
   - EOS markers added at grid boundaries to indicate actual grid size

3. **Flattening**:
   - 30×30 grid → 900-length sequence
   - Both input and output grids processed identically

#### Example:
```
Original 3×3 grid:        Padded 30×30 grid (showing top-left 5×5):
[[7, 0, 7],               [[9, 2, 9, 1, 0],      # values shifted +2
 [7, 0, 7],                [9, 2, 9, 1, 0],      # EOS (1) marks boundary
 [7, 7, 0]]                [9, 9, 2, 1, 0],
                           [1, 1, 1, 0, 0],      # EOS row
                           [0, 0, 0, 0, 0]]      # PAD

Flattened sequence: [9,2,9,1,0,9,2,9,1,0,9,9,2,1,0,1,1,1,0,0,0,0,0,...] (length 900)
```

### 2. Data Augmentation
Located in: `dataset/build_arc_dataset.py:aug()`

- **Dihedral transformations**: 8 variants (rotations and reflections)
- **Color permutations**: Random permutation of colors 1-9 (black/0 excluded)
- **Translational augmentation**: Random padding position (training only)
- Default: 1000 augmentations per puzzle

### 3. Grouping
- Examples from the same puzzle (including augmentations) are grouped together
- Each puzzle assigned a unique `puzzle_identifier` (integer ID)
- Identifiers used for puzzle-specific embeddings

---

## Model Input Shape

### Batch Structure
Located in: `puzzle_dataset.py:PuzzleDataset`

Each batch contains:
```python
{
  "inputs":             # Shape: (batch_size, seq_len)
                        # Type: int32
                        # Values: 0-11 (PAD, EOS, digits 0-9)
                        # seq_len = 900 (30×30)

  "labels":             # Shape: (batch_size, seq_len)
                        # Type: int32
                        # Values: 0-11 or IGNORE_LABEL_ID (-100)
                        # seq_len = 900

  "puzzle_identifiers": # Shape: (batch_size,)
                        # Type: int32
                        # Values: 0 to num_puzzle_identifiers-1
                        # 0 = blank/padding, 1+ = actual puzzles
}
```

### Key Points:
- **batch_size**: Configurable (e.g., 64, 128)
- **seq_len**: Fixed at 900 (30×30 grid flattened)
- **vocab_size**: 12 (PAD + EOS + 10 digits)
- Examples from the same puzzle share the same `puzzle_identifier`

---

## Model Architecture

### TRM (Tiny Recursive Model)
Located in: `models/recursive_reasoning/trm.py:TinyRecursiveReasoningModel_ACTV1`

#### Key Components:

**1. Input Embeddings**
```
Input tokens (batch_size, seq_len=900)
  ↓ [Token Embedding]
Embedded tokens (batch_size, seq_len, hidden_size)
  ↓ [Optional: Concatenate puzzle embeddings]
Embedded (batch_size, seq_len + puzzle_emb_len, hidden_size)
```

- Token embeddings: Learned embeddings for vocab (size 12)
- Puzzle embeddings: Optional learned embeddings per puzzle ID
- Position embeddings: RoPE (Rotary) or Learned

**2. Recursive Reasoning States**

Two latent states maintained across iterations:
```python
z_H: (batch_size, seq_len + puzzle_emb_len, hidden_size)  # High-level state
z_L: (batch_size, seq_len + puzzle_emb_len, hidden_size)  # Low-level state
```

Initialized to learned parameters (H_init, L_init) at the start of each puzzle.

**3. Recursive Update Process**

For `H_cycles` iterations (e.g., 3):
  - For `L_cycles` iterations (e.g., 4):
    - `z_L = L_level(z_L, z_H + input_embeddings)`
  - `z_H = L_level(z_H, z_L)`

Where `L_level` is a stack of transformer blocks (or MLP layers if `mlp_t=True`):
- Self-attention (or MLP on sequence dimension)
- Feed-forward (SwiGLU)
- RMS normalization

**4. Output Heads**

```python
# Language modeling head
logits = lm_head(z_H)[:, puzzle_emb_len:]
# Shape: (batch_size, seq_len=900, vocab_size=12)

# Q-learning head (for adaptive computation)
q_logits = q_head(z_H[:, 0])
# Shape: (batch_size, 2)  [halt, continue]
```

#### Model Configuration (Example: ARC-AGI):
```python
{
  "vocab_size": 12,
  "seq_len": 900,
  "hidden_size": 512,         # embedding dimension
  "num_heads": 8,
  "expansion": 4.0,           # MLP expansion factor
  "H_cycles": 3,              # high-level iterations
  "L_cycles": 4,              # low-level iterations per H
  "L_layers": 2,              # transformer blocks in L_level
  "puzzle_emb_ndim": 8192,    # puzzle embedding dimension
  "halt_max_steps": 1         # ACT max steps (1 = no ACT)
}
```

Total parameters: ~7M

---

## Model Output Shape

### Output Logits (Always Fixed Size)
**Yes, the model ALWAYS predicts full 30×30×12 logits!**

```python
{
  "logits":             # Shape: (batch_size, 900, 12)
                        # - 900 = 30×30 flattened (ALWAYS this size)
                        # - 12 = vocab size (PAD, EOS, digits 0-9)
                        # - Predicted token probabilities for ALL positions

  "q_halt_logits":      # Shape: (batch_size,)
                        # Q-value for halting (ACT)

  "q_continue_logits":  # Shape: (batch_size,)
                        # Q-value for continuing (ACT)
}
```

**Important**:
- The model architecture has a **fixed output size** of 900×12, regardless of actual grid size
- Small grids (e.g., 3×3) still produce 900 predictions, but only the first 9 positions matter
- Padding positions and positions beyond EOS markers are ignored during loss computation
- The actual output grid size is determined during **post-processing**, not during model forward pass

### Post-Processing (Cropping to Actual Size)
Located in: `evaluators/arc.py:_crop()`

After getting logits from the model:
1. **Argmax**: Get predicted tokens from logits: `preds = argmax(logits, dim=-1)` → shape (batch_size, 900)
2. **Reshape**: Reshape to 2D grid: `grid = reshape(preds, (30, 30))` → shape (30, 30)
3. **Crop**: Find the largest rectangle without EOS (token=1) inside
4. **Unmap values**: Subtract 2 to get back to 0-9 range
5. **Result**: Variable-sized output grid (e.g., 3×3, 9×9, 15×20, etc.)

---

## How Test Cases Work

### Original ARC-AGI Puzzle Structure
Each ARC-AGI puzzle consists of:
- **Train examples**: 3-5 input-output pairs demonstrating a transformation pattern
- **Test examples**: 1+ input grids where the output must be predicted using the learned pattern

Example:
```json
{
  "007bbfb7": {
    "train": [
      {"input": [[0,7,7], [7,7,7], ...], "output": [[0,0,0], [7,7,7], ...]},
      {"input": [[7,0,7], ...], "output": [[7,7,7], ...]},
      {"input": [[...]], "output": [[...]]},
      {"input": [[...]], "output": [[...]]},
      {"input": [[...]], "output": [[...]]}
    ],
    "test": [
      {"input": [[7,0,7], [7,0,7], ...]}
    ]
  }
}
```

### Dataset Splitting in This Codebase
Located in: `dataset/build_arc_dataset.py:load_puzzles_arcagi()`

The build process splits ARC examples into training and test datasets:

```python
# Line 168-172
train_examples_dest = ("train", "all")     # ARC "train" → dataset "train" split
test_examples_dest  = ("test", "all")      # ARC "test" → dataset "test" split
```

**Result after `build_arc_dataset.py`:**
```
data/arc1concept-aug-1000/
  train/                         ← TRAINING DATASET
    all__inputs.npy              # Contains 5 train example INPUTS
    all__labels.npy              # Contains 5 train example OUTPUTS
    all__puzzle_identifiers.npy  # All 5 have same puzzle_id (e.g., 123)

  test/                          ← SEPARATE TEST DATASET
    all__inputs.npy              # Contains 1 test example INPUT
    all__labels.npy              # Contains 1 test example OUTPUT (for eval)
    all__puzzle_identifiers.npy  # Same puzzle_id=123 as train examples
```

**Critical Points**:
- ✅ **Train and test ARE split into separate dataset folders**
- ✅ Each example is stored as a **separate data point**, NOT concatenated together!
- ✅ During training, the model ONLY sees examples from the `train/` folder
- ✅ During inference, the model processes examples from the `test/` folder
- ✅ Both splits share the same `puzzle_identifier` for the same puzzle

### Training Phase
1. **Individual examples**: Each of the 5 train examples becomes a separate batch item
   - Batch item 1: `input_1 (900 tokens)` → `output_1 (900 tokens)`, puzzle_id=123
   - Batch item 2: `input_2 (900 tokens)` → `output_2 (900 tokens)`, puzzle_id=123
   - Batch item 3: `input_3 (900 tokens)` → `output_3 (900 tokens)`, puzzle_id=123
   - etc.

2. **Not in-context learning**: The model does NOT see multiple examples in a single forward pass like:
   ```
   ❌ WRONG: [input_1, output_1, input_2, output_2, input_3, output_3, input_test, ???]
   ```

3. **Learning mechanism**:
   - All examples share the same `puzzle_identifier` (e.g., 123)
   - The model learns a **puzzle-specific embedding** for ID=123
   - Over many training iterations, it learns: "puzzle_id=123 → transformation rule T"
   - The puzzle embedding stores the learned transformation pattern

### Inference Phase
1. **Given**: Test input grid from a puzzle seen during training
2. **Process**:
   - Convert test input grid to 900-token sequence
   - Use the **same puzzle_identifier** as the train examples (e.g., 123)
   - Model retrieves the learned embedding for puzzle_id=123
   - Applies the learned transformation via recursive reasoning
   - Outputs predicted sequence (900 tokens)
   - Decode sequence back to output grid

3. **Result**: Predicted output grid

### Example Workflow
```
Training (over many epochs):
  Batch step 15:  Puzzle #123, train_ex_1: input_1 → output_1  (puzzle_id=123)
  Batch step 73:  Puzzle #123, train_ex_2: input_2 → output_2  (puzzle_id=123)
  Batch step 142: Puzzle #123, train_ex_3: input_3 → output_3  (puzzle_id=123)
  Batch step 205: Puzzle #123, train_ex_4: input_4 → output_4  (puzzle_id=123)
  Batch step 381: Puzzle #123, train_ex_5: input_5 → output_5  (puzzle_id=123)
  ...
  → Model gradually learns: "puzzle_id=123 has transformation pattern T"
  → This pattern is encoded in the learned embedding for puzzle_id=123

Inference:
  Test input: Puzzle #123, test_ex: input_test → ???  (puzzle_id=123)
  → Model loads embedding for puzzle_id=123
  → Applies learned transformation T via recursive reasoning
  → Outputs: predicted_output_test
```

### Key Insight
The model doesn't learn a single universal transformation function. Instead:
- It learns **puzzle-specific transformations** stored in puzzle embeddings
- Each puzzle_identifier acts as a key to retrieve the learned transformation pattern
- The recursive reasoning process applies and refines the transformation iteratively
- Training examples from the same puzzle teach the model the pattern through their shared ID
- **This is NOT few-shot in-context learning** - it's learning via puzzle-specific parameters

---

## Loss Function

Located in: `models/losses.py` (assumed, referenced as `IGNORE_LABEL_ID`)

### Cross-Entropy Loss
```python
loss = CrossEntropyLoss(
  input=logits,           # (batch_size, seq_len, vocab_size)
  target=labels,          # (batch_size, seq_len)
  ignore_index=-100       # IGNORE_LABEL_ID for padding
)
```

- Computed per-token across the sequence
- PAD tokens (label=0) typically set to IGNORE_LABEL_ID to exclude from loss
- Only positions corresponding to actual output grid contribute to loss

### ACT Q-Learning Loss (Optional)
When `halt_max_steps > 1`, additional Q-learning loss for adaptive computation:
- Learns when to halt recursive reasoning
- Based on Q(halt) vs Q(continue) values
- Exploration via random minimum halt steps during training

---

## Summary Table

| Component | Shape | Description |
|-----------|-------|-------------|
| **Input Grid** | Variable (≤30×30) | Original 2D grid with values 0-9 |
| **Input Sequence** | (batch, 900) | Flattened, padded, value-shifted grid |
| **Input Embeddings** | (batch, 900+emb_len, hidden) | Embedded input tokens + puzzle emb |
| **Latent States** | (batch, 900+emb_len, hidden) | z_H and z_L recursive states |
| **Output Logits** | (batch, 900, 12) | Predicted token probabilities |
| **Output Sequence** | (batch, 900) | Argmax of logits |
| **Output Grid** | Variable (≤30×30) | Decoded, cropped, unmapped grid |

---

## References

- Paper: "Less is More: Recursive Reasoning with Tiny Networks" ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871))
- Code structure:
  - Data: `dataset/build_arc_dataset.py`, `puzzle_dataset.py`
  - Model: `models/recursive_reasoning/trm.py`
  - Training: `pretrain.py`
