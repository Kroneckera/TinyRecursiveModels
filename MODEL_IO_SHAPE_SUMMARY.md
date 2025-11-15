# TRM Model Input/Output Shape Summary

## Overview
This document explains the input/output shapes of the Tiny Recursive Model (TRM) and how it processes test cases consisting of multiple input-output example pairs.

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

### Training Output
```python
{
  "logits":             # Shape: (batch_size, seq_len=900, vocab_size=12)
                        # Predicted token probabilities for output grid

  "q_halt_logits":      # Shape: (batch_size,)
                        # Q-value for halting (ACT)

  "q_continue_logits":  # Shape: (batch_size,)
                        # Q-value for continuing (ACT)
}
```

### Inference Output
The model outputs predicted tokens which are then:
1. Reshaped to 30×30 grid
2. Value-unmapped (subtract 2, clip negatives to 0)
3. Cropped to actual output size (determined by EOS markers or expected size)

---

## How Test Cases Work

### Training Phase
1. **Multiple examples per puzzle**: The model sees several (input, output) pairs from the same puzzle during training
2. **Shared puzzle identifier**: All examples from puzzle X share identifier ID=X
3. **Puzzle embeddings**: The model learns a unique embedding for each puzzle ID
4. **Pattern learning**: By seeing multiple examples with the same puzzle_id, the model learns the transformation rule for that puzzle

### Inference Phase
1. **Given**: New input grid from a known puzzle type
2. **Process**:
   - Convert input grid to sequence (as in training)
   - Use the same puzzle_identifier as training examples
   - Model applies learned transformation
   - Decode output sequence back to grid
3. **Result**: Predicted output grid

### Example Workflow
```
Training:
- Puzzle #123, Example 1: input_1 → output_1  (puzzle_id=123)
- Puzzle #123, Example 2: input_2 → output_2  (puzzle_id=123)
- Puzzle #123, Example 3: input_3 → output_3  (puzzle_id=123)
  → Model learns: puzzle_id=123 corresponds to transformation rule T

Inference:
- Puzzle #123, Test: input_test → ???  (puzzle_id=123)
  → Model applies learned rule T
  → Outputs: predicted_output_test
```

### Key Insight
The model doesn't learn a single universal transformation function. Instead:
- It learns **puzzle-specific transformations** via puzzle embeddings
- Each puzzle_identifier acts as a key to retrieve the learned transformation rule
- The recursive reasoning process applies the transformation iteratively to refine the output

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
