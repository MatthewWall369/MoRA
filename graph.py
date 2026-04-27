# %%
import os
import re
import time
import urllib.request
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Match dcb.py defaults
SEQ_LEN = 128
EMBED_DIM = 32
HIDDEN = 64
BATCH = 64
VOCAB = 256
LR = 3e-3

# Full-data training knobs
TRAIN_EPOCHS = 5
SAVE_PLOT = True
PLOT_PREFIX = "dcb_benchmark"
DATA_DIR = "gutenberg_data"
CORPUS_PATH = os.path.join(DATA_DIR, "gutenberg_corpus_100.txt")
META_PATH = os.path.join(DATA_DIR, "gutenberg_corpus_100_meta.txt")
MIN_GUTENBERG_BOOKS = 100
GUTENBERG_START_ID = 100
GUTENBERG_MAX_ID = 30000

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DeltaBlock(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)

    def call(self, h, training=None):
        seq_len = tf.shape(h)[1]
        x_prev = tf.pad(h[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        delta = h - x_prev
        counts = tf.cast(tf.range(1, seq_len + 1), tf.float32)
        counts = tf.reshape(counts, [1, -1, 1])
        h_m = tf.nn.sigmoid(delta) * h
        m = tf.cumsum(h_m, axis=1) / counts
        output = m - delta
        return h - output


class CumprodBoundaryLayer(keras.layers.Layer):
    def __init__(self, embed_dim, hidden, **kwargs):
        super().__init__(**kwargs)
        self.bconv = keras.layers.Conv1D(embed_dim, 7, padding="causal", activation="sigmoid")
        self.proj = keras.layers.Dense(hidden, activation="gelu")
        self.norm = keras.layers.LayerNormalization()

    def call(self, e):
        b = self.bconv(e)
        cp = tf.math.cumprod(b, axis=1) * e
        out = self.norm(cp)
        return self.proj(out)


class CumprodLM(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = keras.layers.Embedding(VOCAB, EMBED_DIM)
        self.boundary = CumprodBoundaryLayer(EMBED_DIM, HIDDEN)
        self.mix = keras.layers.Conv1D(HIDDEN, 3, padding="causal", activation="gelu")
        self.norm1 = keras.layers.LayerNormalization()
        self.delta = DeltaBlock(HIDDEN)
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(HIDDEN * 4, activation="gelu"),
                keras.layers.Dense(HIDDEN),
            ]
        )
        self.norm2 = keras.layers.LayerNormalization()
        self.out = keras.layers.Dense(VOCAB)

    def call(self, x):
        e = self.embed(x)
        h = self.boundary(e)
        h = self.mix(h)
        h = self.delta(h)
        h = self.norm1(h)
        h = h + self.mlp(h)
        h = self.norm2(h)
        return self.out(h)


def _download_text(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    return payload.decode("utf-8", errors="replace")


def _strip_gutenberg_boilerplate(text):
    start_re = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
    end_re = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
    start_match = start_re.search(text)
    end_match = end_re.search(text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)
    core = text[start_idx:end_idx].strip()
    return core if len(core) > 1000 else text.strip()


def _fetch_gutenberg_book(book_id):
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in candidates:
        try:
            raw = _download_text(url)
            clean = _strip_gutenberg_boilerplate(raw)
            if len(clean) >= 5000:
                return clean
        except Exception:
            continue
    return None


def ensure_data(min_books=MIN_GUTENBERG_BOOKS):
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CORPUS_PATH) and os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                ids = [int(x) for x in f.read().strip().split(",") if x.strip()]
            if len(ids) >= min_books:
                text = open(CORPUS_PATH, encoding="utf-8").read()
                print(f"Loaded cached Gutenberg corpus: {len(ids)} books, {len(text):,} chars")
                data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
                cut = int(len(data) * 0.9)
                return data, cut
        except Exception:
            pass

    print(f"Collecting at least {min_books} Gutenberg books...")
    books = []
    ids = []
    for book_id in range(GUTENBERG_START_ID, GUTENBERG_MAX_ID + 1):
        if len(books) >= min_books:
            break
        text = _fetch_gutenberg_book(book_id)
        if text is None:
            continue
        books.append(text)
        ids.append(book_id)
        if len(books) % 10 == 0 or len(books) == 1:
            print(f"  collected {len(books):3d}/{min_books} books (last id: {book_id})")

    if len(books) < min_books:
        raise RuntimeError(
            f"Only collected {len(books)} Gutenberg books (needed {min_books}). "
            "Try increasing GUTENBERG_MAX_ID."
        )

    corpus = "\n\n".join(books)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        f.write(corpus)
    with open(META_PATH, "w", encoding="utf-8") as f:
        f.write(",".join(str(i) for i in ids))
    print(f"Saved Gutenberg corpus: {len(books)} books, {len(corpus):,} chars")

    data = np.frombuffer(corpus.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    cut = int(len(data) * 0.9)
    return data, cut


def make_batch(data, lo, hi, batch_size, seq_len, rng):
    starts = rng.integers(lo, hi - seq_len - 1, size=batch_size)
    x = np.stack([data[s : s + seq_len] for s in starts], axis=0).astype(np.int32)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts], axis=0).astype(np.int32)
    return x, y


def make_batch_from_starts(data, starts, seq_len):
    x = np.stack([data[s : s + seq_len] for s in starts], axis=0).astype(np.int32)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts], axis=0).astype(np.int32)
    return x, y


class WindowSequence(keras.utils.Sequence):
    def __init__(self, data, starts, batch_size, seq_len, seed=0, shuffle=False):
        self.data = data
        self.starts = np.asarray(starts, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.starts), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return len(self.starts) // self.batch_size

    def __getitem__(self, idx):
        begin = idx * self.batch_size
        end = begin + self.batch_size
        idxs = self.order[begin:end]
        batch_starts = self.starts[idxs]
        return make_batch_from_starts(self.data, batch_starts, self.seq_len)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


class TrainTimingLogger(keras.callbacks.Callback):
    def __init__(self, batch_size, seq_len, print_every=200):
        super().__init__()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.print_every = max(1, int(print_every))
        self.batch_times_ms = []
        self.batch_losses = []
        self.epoch_mean_ms = []
        self.epoch_tokens_per_sec = []
        self._epoch_times = []
        self._batch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_times = []
        total_epochs = int(self.params.get("epochs", 0))
        steps = self.params.get("steps")
        print(f"\nEpoch {epoch + 1}/{total_epochs} - {steps} full-data train steps")

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        dt_ms = (time.perf_counter() - self._batch_start) * 1000.0
        self.batch_times_ms.append(dt_ms)
        self._epoch_times.append(dt_ms)
        if logs and "loss" in logs:
            self.batch_losses.append(float(logs["loss"]))
        if (batch + 1) % self.print_every == 0:
            loss_val = float(logs.get("loss", np.nan)) if logs else np.nan
            print(f"  step {batch + 1:5d} | loss {loss_val:.4f} | {dt_ms:.2f} ms")

    def on_epoch_end(self, epoch, logs=None):
        if not self._epoch_times:
            return
        mean_ms = float(np.mean(self._epoch_times))
        self.epoch_mean_ms.append(mean_ms)
        tps = (self.batch_size * self.seq_len) / (mean_ms / 1000.0)
        self.epoch_tokens_per_sec.append(float(tps))


def benchmark_device(device_name, data, cut, epochs, seed):
    train_starts = np.arange(0, cut - SEQ_LEN - 1, SEQ_LEN, dtype=np.int64)
    val_starts = np.arange(cut, len(data) - SEQ_LEN - 1, SEQ_LEN, dtype=np.int64)

    train_steps = len(train_starts) // BATCH
    val_steps = len(val_starts) // BATCH
    if train_steps < 1:
        raise RuntimeError("Not enough training windows for one full batch.")
    if val_steps < 1:
        raise RuntimeError("Not enough validation windows for one full batch.")

    train_seq = WindowSequence(
        data=data,
        starts=train_starts[: train_steps * BATCH],
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        seed=seed,
        shuffle=True,
    )
    val_seq = WindowSequence(
        data=data,
        starts=val_starts[: val_steps * BATCH],
        batch_size=BATCH,
        seq_len=SEQ_LEN,
        seed=seed + 1,
        shuffle=False,
    )

    with tf.device(device_name):
        model = CumprodLM()
        model.compile(
            optimizer=keras.optimizers.Adam(LR),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        timing_cb = TrainTimingLogger(BATCH, SEQ_LEN, print_every=200)
        hist = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=epochs,
            verbose=2,
            callbacks=[timing_cb],
        )

    epoch_train_loss = np.array(hist.history.get("loss", []), dtype=np.float32)
    epoch_val_loss = np.array(hist.history.get("val_loss", []), dtype=np.float32)
    epoch_val_acc = np.array(hist.history.get("val_accuracy", []), dtype=np.float32)
    step_ms = np.array(timing_cb.batch_times_ms, dtype=np.float32)
    losses = np.array(timing_cb.batch_losses, dtype=np.float32)
    epoch_mean_ms = np.array(timing_cb.epoch_mean_ms, dtype=np.float32)
    epoch_tokens_per_sec = np.array(timing_cb.epoch_tokens_per_sec, dtype=np.float32)

    mean_ms = float(np.mean(step_ms))
    toks_per_sec = float(np.mean(epoch_tokens_per_sec))
    return {
        "device": device_name,
        "losses": losses,
        "step_ms": step_ms,
        "epoch_train_loss": epoch_train_loss,
        "epoch_mean_ms": epoch_mean_ms,
        "epoch_tokens_per_sec": epoch_tokens_per_sec,
        "epoch_val_loss": epoch_val_loss,
        "epoch_val_acc": epoch_val_acc,
        "mean_step_ms": mean_ms,
        "p50_ms": float(np.percentile(step_ms, 50)),
        "p90_ms": float(np.percentile(step_ms, 90)),
        "tokens_per_sec": toks_per_sec,
        "val_loss": float(np.mean(epoch_val_loss)),
        "val_bpb": float(np.mean(epoch_val_loss) / np.log(2.0)),
        "val_acc": float(np.mean(epoch_val_acc)),
        "train_steps": train_steps,
        "val_steps": val_steps,
    }


def moving_average(x, k=5):
    if len(x) < k:
        return x
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, w, mode="valid")


data_int, train_cut = ensure_data(min_books=MIN_GUTENBERG_BOOKS)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    devices = ["/GPU:0"]
    print(f"Using GPU training on {len(gpus)} detected GPU(s).")
else:
    devices = ["/CPU:0"]
    print("No GPU detected; running training on CPU.")

results = []
for idx, dev in enumerate(devices):
    print(f"Running full-data training on {dev} ...")
    results.append(
        benchmark_device(
            device_name=dev,
            data=data_int,
            cut=train_cut,
            epochs=TRAIN_EPOCHS,
            seed=1234 + idx,
        )
    )

print("\nPerformance summary")
for r in results:
    print(
        f"{r['device']:>7} | "
        f"{r['train_steps']} train steps/epoch | "
        f"{r['val_steps']} val steps/epoch | "
        f"mean {r['mean_step_ms']:.2f} ms/step | "
        f"p90 {r['p90_ms']:.2f} ms | "
        f"{r['tokens_per_sec']:.0f} tok/s | "
        f"val_loss {r['val_loss']:.4f} | "
        f"val_bpb {r['val_bpb']:.4f} | "
        f"val_acc {r['val_acc']:.4f}"
    )

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

epoch_axis = np.arange(1, TRAIN_EPOCHS + 1)

for r in results:
    axes[0, 0].plot(epoch_axis, r["epoch_train_loss"], marker="o", linewidth=1.7, label=r["device"])
    axes[0, 0].plot(epoch_axis, r["epoch_val_loss"], linestyle="--", linewidth=1.5, label=f"{r['device']} val")
axes[0, 0].set_title("Loss Across Epochs")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(alpha=0.25)
axes[0, 0].legend()

for r in results:
    sm = moving_average(r["losses"], k=5)
    x = np.arange(1, len(sm) + 1)
    axes[0, 1].plot(x, sm, label=r["device"])
axes[0, 1].set_title("Train Loss (5-step MA, all steps)")
axes[0, 1].set_xlabel("Global Train Step")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend()

for r in results:
    axes[1, 0].plot(epoch_axis, r["epoch_mean_ms"], marker="o", linewidth=1.8, label=r["device"])
axes[1, 0].set_title("Mean Step Latency Across Epochs")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("ms / step")
axes[1, 0].grid(alpha=0.25)
axes[1, 0].legend()

for r in results:
    axes[1, 1].plot(epoch_axis, r["epoch_tokens_per_sec"], marker="o", linewidth=1.8, label=r["device"])
axes[1, 1].set_title("Throughput Across Epochs")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("tokens / sec")
axes[1, 1].grid(alpha=0.25)
axes[1, 1].legend()

plt.tight_layout()
if SAVE_PLOT:
    out_name = f"{PLOT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out_path = os.path.abspath(out_name)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved benchmark plot: {out_path}")
plt.show()
