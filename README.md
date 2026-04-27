# MoraV5: GPU ByteLM on 100 Gutenberg Books

Welcome to the tiny-but-spicy language model lab.

This repo trains a byte-level LM (`CumprodLM`) over a large Gutenberg corpus and pushes full-data GPU training while logging speed, loss, and validation behavior. It is compact enough to experiment quickly, but weird enough architecturally to be interesting.

## What This Is

- A character/byte next-token prediction model (`VOCAB=256` bytes).
- Trained on **100 Gutenberg books** (cached locally).
- Full-pass epoch training (`model.fit(...)`) over the dataset, not tiny random-step toy loops.
- GPU-first execution with benchmark-style metrics and plots.

## Dataset: Big Text Energy

- Source: Project Gutenberg text files collected automatically.
- Latest cached corpus: **68,493,179 characters** from **100 books**.
- Split: 90% train / 10% validation.
- Windowing: fixed-length byte windows of `SEQ_LEN=128`.
- Objective: predict byte `t+1` from bytes up to `t`.

In other words: this is pure autoregressive byte modeling, no tokenizer, no BPE, no mercy.

## Model Architecture (Nerd Mode)

`CumprodLM` is intentionally non-standard. The stack:

1. **Embedding**
   - `Embedding(256, 32)`
   - Maps raw bytes to a dense latent space.

2. **CumprodBoundaryLayer**
   - Causal `Conv1D` produces a per-position gate-like boundary signal.
   - `cumprod` across time builds multiplicative continuity dynamics.
   - Intuition: the model can represent "keep flowing" vs "new segment / reset-like behavior" in a smooth differentiable way.

3. **Causal Mixing Conv**
   - `Conv1D(HIDDEN, kernel=3, causal)` to blend local temporal context.

4. **DeltaBlock**
   - Computes temporal differences (`delta`) against lagged state.
   - Mixes cumulative behavior with change signals.
   - Intuition: separate stable context from local transitions, then fuse them.

5. **Residual MLP + Norm**
   - Standard feedforward refinement in residual form.

6. **Output Head**
   - Dense projection to 256 logits (one per byte value).

Parameter snapshot from `summary.txt`:

- Trainable params: **79,904**
- Total params shown (including optimizer state): **239,714**

## Training Setup

- Device: **GPU** (`/GPU:0` detected and used in latest run)
- Batch size: **64**
- Sequence length: **128**
- Epochs: **5**
- Optimizer: Adam (`lr=3e-3`)
- Loss: sparse categorical cross-entropy from logits
- Metrics: accuracy + latency/throughput tracking

Per epoch (latest corpus split):

- Train steps: **7,597**
- Validation steps: **844**

## Benchmark Highlights (Latest Log)

From `log.txt`:

- Mean latency: **2.95 ms/step**
- p90 latency: **2.86 ms**
- Throughput: **2,832,502 tokens/sec**
- Aggregate validation loss: **1.7911**
- Validation bits-per-byte: **2.5841**
- Validation accuracy: **0.4843**

Epoch trend:

- Train loss: **1.8378 -> 1.7086**
- Train accuracy: **0.4400 -> 0.4702**
- Best val loss around epoch 4: **1.7708**
- Epoch 5 val loss rises slightly: **1.7891**
- Val accuracy plateaus near: **0.4868**

Interpretation: optimization is healthy and fast; generalization gains start to flatten by epoch 4-5 at current LR/model size.

## Why This Is Cool

- You get strong throughput on a real multi-book corpus with a tiny model.
- Byte-level modeling avoids tokenizer complexity.
- The boundary/cumprod + delta dynamics are unusual and worth experimenting with.
- Training is now full-data and visibly instrumented, so runs are auditable.

## Known Warning

Keras warns that `WindowSequence` should call `super().__init__(**kwargs)`.
Training still works, but patching this is a good cleanup item for stricter Keras compatibility.

## Run It

From project root:

```bash
python graph.py
```

The script will:

1. Load cached corpus or collect Gutenberg books if needed
2. Train on full data with GPU when available
3. Print live step/epoch progress and losses
4. Save a benchmark plot as `dcb_benchmark_YYYYMMDD_HHMMSS.png`

## Next Nerdy Experiments

- Add cosine decay or one-cycle LR schedule.
- Add early stopping around validation loss minima.
- Scale `HIDDEN` and `EMBED_DIM` to map throughput/quality tradeoffs.
- Compare this architecture against a tiny Transformer baseline under equal parameter count.
