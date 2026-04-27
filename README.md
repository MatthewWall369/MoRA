# MoraV5 Gutenberg Training Benchmark

This project trains a byte-level language model (`CumprodLM`) on a Gutenberg corpus and benchmarks full-data training performance on GPU.

## Dataset

- Source: Project Gutenberg text files (auto-collected and cached).
- Corpus size: **100 books**, **68,493,179 characters**.
- Split: 90% train / 10% validation.
- Sequence format: byte-level windows (`SEQ_LEN=128`) with next-byte targets.

## Model

The current model in `graph.py` uses:

- Byte embedding (`VOCAB=256`, `EMBED_DIM=32`)
- `CumprodBoundaryLayer` + causal `Conv1D` mixing
- `DeltaBlock` temporal transform
- Residual MLP block
- Output projection to 256 byte logits

From the latest model summary:

- Trainable params: **79,904**
- Total params (including optimizer state): **239,714**

## Training Configuration

- Device: **GPU** (`/GPU:0` detected and used)
- Batch size: **64**
- Sequence length: **128**
- Epochs: **5**
- Optimizer: Adam (`lr=3e-3`)
- Loss: Sparse categorical cross-entropy (from logits)
- Training mode: full-data epoch traversal using `model.fit(...)`

Per-epoch workload:

- Train steps: **7,597**
- Validation steps: **844**

## Benchmark Results (Latest Run)

From `log.txt`:

- Mean step latency: **2.95 ms/step**
- p90 step latency: **2.86 ms**
- Throughput: **2,832,502 tokens/sec**
- Final aggregate validation loss: **1.7911**
- Validation bits-per-byte (bpb): **2.5841**
- Validation accuracy: **0.4843**

Observed epoch-level trend:

- Train loss improved from **1.8378 -> 1.7086**
- Train accuracy improved from **0.4400 -> 0.4702**
- Validation loss reached best around epoch 4 (**1.7708**), then slightly rose at epoch 5 (**1.7891**)
- Validation accuracy plateaued near **0.4868**

This suggests steady optimization with mild overfitting or noise beyond epoch 4 at current hyperparameters.

## Output Artifacts

- Benchmark plot saved as: `dcb_benchmark_YYYYMMDD_HHMMSS.png`
- Corpus cache files stored under `gutenberg_data/`

## Noted Warning

Keras reported:

- `PyDataset` (`WindowSequence`) should call `super().__init__(**kwargs)` in its constructor.

This warning does not stop training, but updating the sequence constructor is recommended for cleaner Keras integration.

## How To Run

From the project root:

```bash
python graph.py
```

The script will:

1. Load cached Gutenberg corpus (or collect books if cache is missing)
2. Train on full data using GPU when available
3. Print step/epoch progress
4. Save and display benchmark plots
