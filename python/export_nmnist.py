"""
export_nmnist.py - Train/export a sparse N-MNIST benchmark to the RTL format.

N-MNIST is an event-camera version of MNIST. This exporter trains a small
count-based MLP on binned event streams, prunes the first layer, quantizes the
weights, converts the model to the same event-driven SNN format as the RTL, and
writes sim/*.mem for the existing Verilator runners.

Prerequisite:
    python3 -m pip install tonic

Usage:
    python3 python/export_nmnist.py
    bash tools/run_verilator_event.sh 0 299 _nmnist
    bash tools/run_verilator_ann.sh 0 299 _nmnist
"""

import json
import os
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(int(os.environ.get("SNN_TORCH_THREADS", "1")))
torch.set_num_interop_threads(int(os.environ.get("SNN_TORCH_INTEROP_THREADS", "1")))


WIDTH = 34
HEIGHT = 34
POLARITIES = 2
N_INPUT = WIDTH * HEIGHT * POLARITIES
N_HIDDEN = int(os.environ.get("NMNIST_N_HIDDEN", "128"))
N_OUTPUT = 10
N_TOTAL = N_INPUT + 1 + N_HIDDEN + N_OUTPUT

ID_BIAS = N_INPUT
ID_HIDDEN_BASE = N_INPUT + 1
ID_OUTPUT_BASE = N_INPUT + 1 + N_HIDDEN

T_STEPS = int(os.environ.get("NMNIST_T_STEPS", "30"))
NUM_TRAIN = int(os.environ.get("NMNIST_NUM_TRAIN", "5000"))
NUM_TEST_RTL = int(os.environ.get("NMNIST_NUM_TEST_RTL", "300"))
EPOCHS = int(os.environ.get("NMNIST_EPOCHS", "18"))
BATCH_SIZE = int(os.environ.get("NMNIST_BATCH_SIZE", "128"))
TOPK_W1 = int(os.environ.get("NMNIST_TOPK_W1", "128"))
FINETUNE_EPOCHS = int(os.environ.get("NMNIST_FINETUNE_EPOCHS", "12"))
BIAS_MODE = os.environ.get("NMNIST_BIAS_MODE", "none").lower()
READOUT = os.environ.get("NMNIST_READOUT", "membrane").lower()
PRUNE_MODE = os.environ.get("NMNIST_PRUNE_MODE", "per_hidden").lower()
MIN_W1_PER_HIDDEN = int(os.environ.get("NMNIST_MIN_W1_PER_HIDDEN", "0"))
ACTIVITY_LAMBDA = float(os.environ.get("NMNIST_ACTIVITY_LAMBDA", "0.0"))
MEMBRANE_WIDTH = int(os.environ.get("NMNIST_MEMBRANE_WIDTH", "16"))
HW_WRAP = os.environ.get("NMNIST_HW_WRAP", "1").lower() not in {"0", "false", "no", "off"}

THR_MIN = int(os.environ.get("NMNIST_THR_MIN", "20"))
THR_MAX = int(os.environ.get("NMNIST_THR_MAX", "1480"))
THR_STEP = int(os.environ.get("NMNIST_THR_STEP", "20"))
THRESHOLD_OBJECTIVE = os.environ.get("NMNIST_THRESHOLD_OBJECTIVE", "accuracy").lower()
THRESHOLD_OP_PENALTY = float(os.environ.get("NMNIST_THRESHOLD_OP_PENALTY", "0.0"))
THR_RANGE = range(THR_MIN, THR_MAX + 1, THR_STEP)
TUNE_SUBSET = int(os.environ.get("NMNIST_TUNE_SUBSET", "500"))
EVAL_SUBSET = int(os.environ.get("NMNIST_EVAL_SUBSET", "500"))

LEAK = 1
SEED = 13
SELECTION = os.environ.get("NMNIST_SELECTION", "balanced")

if BIAS_MODE not in {"none", "hidden", "all"}:
    raise SystemExit("NMNIST_BIAS_MODE must be one of: none, hidden, all")
if READOUT not in {"spike", "membrane"}:
    raise SystemExit("NMNIST_READOUT must be one of: spike, membrane")
if PRUNE_MODE not in {"per_hidden", "global", "saliency"}:
    raise SystemExit("NMNIST_PRUNE_MODE must be one of: per_hidden, global, saliency")
if MIN_W1_PER_HIDDEN < 0:
    raise SystemExit("NMNIST_MIN_W1_PER_HIDDEN must be non-negative")
if THRESHOLD_OBJECTIVE not in {"accuracy", "ops", "edge"}:
    raise SystemExit("NMNIST_THRESHOLD_OBJECTIVE must be one of: accuracy, ops, edge")
if THR_STEP <= 0:
    raise SystemExit("NMNIST_THR_STEP must be positive")
if MEMBRANE_WIDTH <= 1:
    raise SystemExit("NMNIST_MEMBRANE_WIDTH must be greater than 1")


def wrap_signed(values, width: int = MEMBRANE_WIDTH) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    return (((arr & mask) ^ sign) - sign).astype(np.int32)


def hw_mem(values) -> np.ndarray:
    if not HW_WRAP:
        return np.asarray(values, dtype=np.int32)
    return wrap_signed(values, MEMBRANE_WIDTH)


def load_tonic_nmnist():
    try:
        import tonic
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'tonic'. Install it in your active environment:\n"
            "  python3 -m pip install tonic"
        ) from exc

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.environ.get("SNN_DATA_DIR", os.path.join(repo_root, "data"))
    train = tonic.datasets.NMNIST(save_to=data_dir, train=True)
    test = tonic.datasets.NMNIST(save_to=data_dir, train=False)
    return train, test


def event_fields(events) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = events.dtype.names
    if names is None:
        arr = np.asarray(events)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError("Unsupported N-MNIST event array shape")
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    if "x" not in names or "y" not in names or "t" not in names:
        raise ValueError(f"N-MNIST event dtype missing x/y/t fields: {names}")
    polarity = events["p"] if "p" in names else np.zeros(len(events), dtype=np.int64)
    return events["x"], events["y"], events["t"], polarity


def events_to_spikes(events) -> np.ndarray:
    spikes = np.zeros((T_STEPS, N_INPUT), dtype=np.uint8)
    if len(events) == 0:
        return spikes

    x, y, t, p = event_fields(events)
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p, dtype=np.int64)

    valid = (x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)
    if not np.any(valid):
        return spikes
    x = x[valid]
    y = y[valid]
    t = t[valid]
    p = np.clip(p[valid], 0, POLARITIES - 1)

    t0 = float(t.min())
    t1 = float(t.max())
    if t1 <= t0:
        bins = np.zeros_like(x)
    else:
        bins = ((t - t0) * T_STEPS / (t1 - t0 + 1e-12)).astype(np.int64)
        bins = np.clip(bins, 0, T_STEPS - 1)
    channels = p * (WIDTH * HEIGHT) + y * WIDTH + x
    spikes[bins, channels] = 1
    return spikes


def dataset_labels(dataset) -> np.ndarray | None:
    for attr in ("targets", "labels"):
        values = getattr(dataset, attr, None)
        if values is not None:
            arr = np.asarray(values, dtype=np.int64)
            if len(arr) >= len(dataset):
                return arr[:len(dataset)]
    for attr in ("data", "files", "file_paths"):
        values = getattr(dataset, attr, None)
        if values is None or len(values) < len(dataset):
            continue
        labels = []
        for item in values[:len(dataset)]:
            parent = os.path.basename(os.path.dirname(str(item)))
            if not parent.isdigit():
                labels = []
                break
            labels.append(int(parent))
        if labels:
            return np.asarray(labels, dtype=np.int64)
    return None


def interleave_by_class(selected: list[list[int]]) -> list[int]:
    order: list[int] = []
    max_len = max((len(items) for items in selected), default=0)
    for offset in range(max_len):
        for cls_items in selected:
            if offset < len(cls_items):
                order.append(cls_items[offset])
    return order


def select_indices(dataset, limit: int, split_name: str) -> list[int]:
    n = min(limit, len(dataset))
    if SELECTION == "sequential":
        indices = list(range(n))
        print(f"  {split_name} selection: sequential first {len(indices)} samples")
        return indices
    if SELECTION != "balanced":
        raise SystemExit("NMNIST_SELECTION must be 'balanced' or 'sequential'")

    labels = dataset_labels(dataset)
    if labels is None:
        raise SystemExit(
            "Could not read N-MNIST labels without loading events. "
            "Set NMNIST_SELECTION=sequential to use the old ordered subset."
        )

    rng = np.random.default_rng(SEED + (0 if split_name == "train" else 1000))
    per_class = n // N_OUTPUT
    remainder = n % N_OUTPUT
    selected: list[list[int]] = []
    for cls in range(N_OUTPUT):
        cls_indices = np.flatnonzero(labels == cls)
        rng.shuffle(cls_indices)
        quota = per_class + (1 if cls < remainder else 0)
        selected.append([int(i) for i in cls_indices[:quota]])

    indices = interleave_by_class(selected)
    hist = np.bincount(labels[indices], minlength=N_OUTPUT)
    print(
        f"  {split_name} selection: balanced {len(indices)} samples "
        f"(class counts: {' '.join(str(int(v)) for v in hist)})"
    )
    return indices


def load_split(dataset, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(indices)
    spikes = np.zeros((n, T_STEPS, N_INPUT), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    for i, dataset_idx in enumerate(indices):
        events, label = dataset[dataset_idx]
        spikes[i] = events_to_spikes(events)
        labels[i] = int(label)
        if (i + 1) % 250 == 0:
            print(f"  loaded {i + 1}/{n} samples")
    return spikes, labels


def eval_model(model: nn.Module, counts: np.ndarray, labels: np.ndarray) -> float:
    scale = max(T_STEPS, 1)
    x = torch.tensor(counts / scale, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        return (pred == y).float().mean().item()


def train_mlp(train_counts: np.ndarray, train_labels: np.ndarray,
              test_counts: np.ndarray, test_labels: np.ndarray):
    torch.manual_seed(SEED)
    hidden_bias = BIAS_MODE in {"hidden", "all"}
    output_bias = BIAS_MODE == "all"
    model = nn.Sequential(
        nn.Linear(N_INPUT, N_HIDDEN, bias=hidden_bias),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_OUTPUT, bias=output_bias),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    scale = max(T_STEPS, 1)
    x_train = torch.tensor(train_counts / scale, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = torch.tensor(test_counts / scale, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(x_train), generator=gen)
        model.train()
        for start in range(0, len(x_train), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        if epoch in {0, EPOCHS - 1}:
            with torch.no_grad():
                pred = model(x_test).argmax(dim=1)
                acc = (pred == y_test).float().mean().item()
            print(f"  epoch {epoch + 1:02d}/{EPOCHS}: test-count acc {acc * 100:.2f}%")

    train_acc = eval_model(model, train_counts, train_labels)
    test_acc = eval_model(model, test_counts, test_labels)
    return model, train_acc, test_acc


def prune_w1(w1: np.ndarray) -> np.ndarray:
    sparse = np.zeros_like(w1)
    k = min(TOPK_W1, w1.shape[1])
    for h in range(w1.shape[0]):
        idx = np.argsort(np.abs(w1[h]))[-k:]
        sparse[h, idx] = w1[h, idx]
    return sparse


def topk_mask(w1: np.ndarray, train_counts: np.ndarray | None = None,
              w2: np.ndarray | None = None) -> np.ndarray:
    mask = np.zeros_like(w1, dtype=np.float32)
    k = min(TOPK_W1, w1.shape[1])
    if PRUNE_MODE == "per_hidden":
        for h in range(w1.shape[0]):
            idx = np.argsort(np.abs(w1[h]))[-k:]
            mask[h, idx] = 1.0
        return mask

    budget = min(k * w1.shape[0], w1.size)
    scores = np.abs(w1).astype(np.float64)
    if PRUNE_MODE == "saliency":
        if train_counts is not None:
            activity = np.sqrt(np.mean(train_counts, axis=0).astype(np.float64) + 1e-6)
            scores *= activity[np.newaxis, :]
        if w2 is not None:
            hidden_importance = np.linalg.norm(w2, axis=0).astype(np.float64) + 1e-6
            scores *= hidden_importance[:, np.newaxis]

    min_per_hidden = min(MIN_W1_PER_HIDDEN, k)
    if min_per_hidden > 0:
        for h in range(w1.shape[0]):
            idx = np.argsort(scores[h])[-min_per_hidden:]
            mask[h, idx] = 1.0

    selected = int(mask.sum())
    remaining = max(0, budget - selected)
    if remaining > 0:
        flat_scores = scores.copy()
        flat_scores[mask.astype(bool)] = -np.inf
        flat_idx = np.argpartition(flat_scores.ravel(), -remaining)[-remaining:]
        mask.ravel()[flat_idx] = 1.0
    return mask


def finetune_sparse(model: nn.Module, mask_np: np.ndarray,
                    train_counts: np.ndarray, train_labels: np.ndarray,
                    test_counts: np.ndarray, test_labels: np.ndarray):
    if FINETUNE_EPOCHS <= 0:
        return model, eval_model(model, train_counts, train_labels), eval_model(model, test_counts, test_labels)

    scale = max(T_STEPS, 1)
    x_train = torch.tensor(train_counts / scale, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    mask = torch.tensor(mask_np, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(SEED + 77)

    with torch.no_grad():
        model[0].weight.mul_(mask)

    for epoch in range(FINETUNE_EPOCHS):
        perm = torch.randperm(len(x_train), generator=gen)
        model.train()
        for start in range(0, len(x_train), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            hidden = model[1](model[0](x_train[idx]))
            logits = model[2](hidden)
            loss = loss_fn(logits, y_train[idx])
            if ACTIVITY_LAMBDA > 0.0:
                loss = loss + ACTIVITY_LAMBDA * hidden.mean()
            loss.backward()
            model[0].weight.grad.mul_(mask)
            opt.step()
            with torch.no_grad():
                model[0].weight.mul_(mask)
        if epoch in {0, FINETUNE_EPOCHS - 1}:
            acc = eval_model(model, test_counts, test_labels)
            print(f"  sparse finetune {epoch + 1:02d}/{FINETUNE_EPOCHS}: test-count acc {acc * 100:.2f}%")

    return model, eval_model(model, train_counts, train_labels), eval_model(model, test_counts, test_labels)


def quantize_signed_with_scale(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(weights)))
    if max_abs == 0:
        return np.zeros_like(weights, dtype=np.int8), 1.0
    scale = 127.0 / max_abs
    q = np.round(weights * scale)
    return np.clip(q, -127, 127).astype(np.int8), scale


def quantize_signed(weights: np.ndarray) -> np.ndarray:
    return quantize_signed_with_scale(weights)[0]


def quantize_folded(w1_sparse: np.ndarray, w2: np.ndarray,
                    b1: np.ndarray | None = None,
                    b2: np.ndarray | None = None):
    """Per-hidden W1 quantization, with the hidden scale folded into W2.

    The RTL has no floating scale multipliers. For hidden neuron h, qW1[h] is
    scaled independently for better int8 resolution. Dividing W2[:, h] by the
    same scale keeps the output logits equivalent up to one global W2 scale.
    Hidden/output bias is represented in the SNN as a bias neuron that fires
    once per timestep, so the ANN bias memories receive the per-timestep bias
    multiplied by T_STEPS.
    """
    w1q = np.zeros_like(w1_sparse, dtype=np.int8)
    scales = np.ones(w1_sparse.shape[0], dtype=np.float32)
    for h in range(w1_sparse.shape[0]):
        max_abs = float(np.max(np.abs(w1_sparse[h])))
        if max_abs > 0:
            scales[h] = 127.0 / max_abs
            q = np.round(w1_sparse[h] * scales[h])
            w1q[h] = np.clip(q, -127, 127).astype(np.int8)
    w2_folded = w2 / scales[np.newaxis, :]
    w2q, w2_scale = quantize_signed_with_scale(w2_folded)

    b1_spike = np.zeros(N_HIDDEN, dtype=np.int8)
    b2_spike = np.zeros(N_OUTPUT, dtype=np.int8)
    if b1 is not None:
        q = np.round(b1 * scales)
        b1_spike = np.clip(q, -127, 127).astype(np.int8)
    if b2 is not None:
        q = np.round(b2 * w2_scale)
        b2_spike = np.clip(q, -127, 127).astype(np.int8)

    b1_ann = (b1_spike.astype(np.int32) * T_STEPS).astype(np.int32)
    b2_ann = (b2_spike.astype(np.int32) * T_STEPS).astype(np.int32)
    return w1q, w2q, b1_spike, b2_spike, b1_ann, b2_ann


def simulate_ann_int(counts: np.ndarray, labels: np.ndarray, w1q: np.ndarray,
                     w2q: np.ndarray, b1_ann: np.ndarray, b2_ann: np.ndarray,
                     hidden_shift: int) -> float:
    h_raw = counts.astype(np.int32) @ w1q.T.astype(np.int32) + b1_ann
    h = np.maximum(h_raw, 0) >> hidden_shift
    h = np.minimum(h, 255).astype(np.int32)
    y = h @ w2q.T.astype(np.int32) + b2_ann
    pred = np.argmax(y, axis=1)
    return float((pred == labels).mean())


def choose_ann_hidden_shift(counts: np.ndarray, labels: np.ndarray,
                            w1q: np.ndarray, w2q: np.ndarray,
                            b1_ann: np.ndarray, b2_ann: np.ndarray) -> int:
    best_shift, best_acc = 0, -1.0
    for shift in range(16):
        acc = simulate_ann_int(counts, labels, w1q, w2q, b1_ann, b2_ann, shift)
        if acc > best_acc:
            best_shift, best_acc = shift, acc
    return best_shift


def lif_leak(mem: np.ndarray) -> np.ndarray:
    leaked = np.where(mem > LEAK, mem - LEAK, np.where(mem > 0, 0, mem))
    return hw_mem(leaked)


def precompute_hidden_input(spikes: np.ndarray, w1q: np.ndarray,
                            b1_spike: np.ndarray) -> np.ndarray:
    w1i = w1q.astype(np.int32)
    hin = np.zeros((spikes.shape[0], T_STEPS, N_HIDDEN), dtype=np.int32)
    b1i = b1_spike.astype(np.int32)
    for t in range(T_STEPS):
        hin[:, t, :] = hw_mem(spikes[:, t, :].astype(np.int32) @ w1i.T + b1i)
    return hin


def simulate_snn_hin(hin: np.ndarray, labels: np.ndarray,
                     w2q: np.ndarray, b2_spike: np.ndarray,
                     threshold: int, return_stats: bool = False):
    num = hin.shape[0]
    mem_h = np.zeros((num, N_HIDDEN), dtype=np.int32)
    mem_o = np.zeros((num, N_OUTPUT), dtype=np.int32)
    hidden_prev = np.zeros((num, N_HIDDEN), dtype=np.int32)
    out_count = np.zeros((num, N_OUTPUT), dtype=np.int32)
    w2i = w2q.astype(np.int32)
    b2i = b2_spike.astype(np.int32)
    hidden_fanout = np.count_nonzero(w2q, axis=0).astype(np.int32)
    hidden_spikes = 0
    hidden_deliveries = 0
    output_spikes = 0
    threshold_i = int(hw_mem(threshold))

    for t in range(T_STEPS):
        mem_h = hw_mem(lif_leak(mem_h).astype(np.int64) + hin[:, t, :].astype(np.int64))
        hidden = mem_h >= threshold_i
        mem_h[hidden] = 0

        hidden_deliveries += int((hidden_prev * hidden_fanout).sum())
        oin = hw_mem(hidden_prev @ w2i.T + b2i)
        mem_o = hw_mem(lif_leak(mem_o).astype(np.int64) + oin.astype(np.int64))
        if READOUT == "spike":
            out = mem_o >= threshold_i
            mem_o[out] = 0
            out_count += out.astype(np.int32)
            output_spikes += int(out.sum())
        else:
            out = np.zeros_like(mem_o, dtype=bool)
        hidden_spikes += int(hidden.sum())
        hidden_prev = hidden.astype(np.int32)

    pred = np.argmax(out_count if READOUT == "spike" else mem_o, axis=1)
    acc = float((pred == labels).mean())
    if not return_stats:
        return acc
    return acc, {
        "hidden_spikes": hidden_spikes,
        "hidden_deliveries": hidden_deliveries,
        "output_spikes": output_spikes,
    }


def build_csr(w1q: np.ndarray, w2q: np.ndarray,
              b1_spike: np.ndarray, b2_spike: np.ndarray):
    by_src = [[] for _ in range(N_TOTAL)]
    for h in np.flatnonzero(b1_spike):
        by_src[ID_BIAS].append((ID_HIDDEN_BASE + int(h), int(b1_spike[h])))
    for out in np.flatnonzero(b2_spike):
        by_src[ID_BIAS].append((ID_OUTPUT_BASE + int(out), int(b2_spike[out])))
    for h in range(N_HIDDEN):
        hid = ID_HIDDEN_BASE + h
        for src in np.flatnonzero(w1q[h]):
            by_src[int(src)].append((hid, int(w1q[h, src])))
    for out in range(N_OUTPUT):
        out_id = ID_OUTPUT_BASE + out
        for h in np.flatnonzero(w2q[out]):
            by_src[ID_HIDDEN_BASE + int(h)].append((out_id, int(w2q[out, h])))

    csr_dst = []
    csr_weight = []
    src_start = [0] * N_TOTAL
    src_count = [0] * N_TOTAL
    for src in range(N_TOTAL):
        src_start[src] = len(csr_dst)
        for dst, weight in by_src[src]:
            csr_dst.append(dst)
            csr_weight.append(weight)
        src_count[src] = len(by_src[src])
    return csr_dst, csr_weight, src_start, src_count


def write_hex(path: str, values: Iterable[int], width: int):
    mask = (1 << (4 * width)) - 1
    with open(path, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}X}\n")


def estimate_input_deliveries(spikes: np.ndarray, w1q: np.ndarray,
                              b1_spike: np.ndarray, b2_spike: np.ndarray) -> int:
    input_fanout = np.count_nonzero(w1q, axis=0).astype(np.int64)
    input_events = spikes.sum(axis=(0, 1)).astype(np.int64)
    bias_fanout = int(np.count_nonzero(b1_spike) + np.count_nonzero(b2_spike))
    bias_events = int(spikes.shape[0] * T_STEPS * bias_fanout)
    return int(input_events @ input_fanout + bias_events)


def threshold_score(acc: float, total_deliveries: int, num_samples: int) -> float:
    if THRESHOLD_OBJECTIVE == "accuracy":
        return acc
    ops_per_sample = total_deliveries / max(num_samples, 1)
    if THRESHOLD_OBJECTIVE == "ops":
        return acc / max(ops_per_sample, 1.0)
    dense_ops = N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT
    return acc - THRESHOLD_OP_PENALTY * (ops_per_sample / max(dense_ops, 1))


def write_export_metrics(sim_dir: str, metrics: dict):
    with open(os.path.join(sim_dir, "export_nmnist_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def export(sim_dir: str, spikes: np.ndarray, labels: np.ndarray,
           counts: np.ndarray, w1q: np.ndarray, w2q: np.ndarray,
           b1_spike: np.ndarray, b2_spike: np.ndarray,
           b1_ann: np.ndarray, b2_ann: np.ndarray,
           threshold: int, hidden_shift: int):
    os.makedirs(sim_dir, exist_ok=True)
    csr_dst, csr_weight, src_start, src_count = build_csr(w1q, w2q, b1_spike, b2_spike)

    write_hex(os.path.join(sim_dir, "csr_dst.mem"), csr_dst, 4)
    write_hex(os.path.join(sim_dir, "csr_weight.mem"), csr_weight, 2)
    write_hex(os.path.join(sim_dir, "csr_start.mem"), src_start, 6)
    write_hex(os.path.join(sim_dir, "csr_count.mem"), src_count, 6)

    with open(os.path.join(sim_dir, "snn_spikes.mem"), "w") as f:
        for img in spikes:
            for t in range(T_STEPS):
                f.write("".join("1" if img[t, p] else "0"
                                for p in range(N_INPUT - 1, -1, -1)) + "\n")

    with open(os.path.join(sim_dir, "snn_labels.mem"), "w") as f:
        for lab in labels:
            f.write(f"{int(lab)}\n")

    write_hex(os.path.join(sim_dir, "ann_pixels.mem"),
              np.minimum(counts, 255).astype(np.uint8).flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_w1.mem"), w1q.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_b1.mem"), b1_ann, 8)
    write_hex(os.path.join(sim_dir, "ann_w2.mem"), w2q.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_b2.mem"), b2_ann, 8)

    with open(os.path.join(sim_dir, "snn_config.vh"), "w") as f:
        label_hist = ",".join(str(int(v)) for v in np.bincount(labels, minlength=N_OUTPUT))
        f.write("// Auto-generated by python/export_nmnist.py - do not edit by hand\n")
        f.write("`ifndef SNN_CONFIG_VH\n`define SNN_CONFIG_VH\n")
        f.write("`define SNN_DATASET        \"nmnist\"\n")
        f.write(f"`define SNN_SELECTION      \"{SELECTION}\"\n")
        f.write(f"`define SNN_LABEL_HIST     \"{label_hist}\"\n")
        f.write(f"`define SNN_N_TOTAL        {N_TOTAL}\n")
        f.write(f"`define SNN_N_INPUT        {N_INPUT}\n")
        f.write(f"`define SNN_N_HIDDEN       {N_HIDDEN}\n")
        f.write(f"`define SNN_N_OUTPUT       {N_OUTPUT}\n")
        f.write(f"`define SNN_ID_BIAS        {ID_BIAS}\n")
        f.write(f"`define SNN_ID_HIDDEN_BASE {ID_HIDDEN_BASE}\n")
        f.write(f"`define SNN_ID_OUTPUT_BASE {ID_OUTPUT_BASE}\n")
        f.write(f"`define SNN_T_STEPS        {T_STEPS}\n")
        f.write(f"`define SNN_THRESHOLD      {threshold}\n")
        f.write(f"`define SNN_LEAK           {LEAK}\n")
        f.write(f"`define SNN_NUM_TEST       {len(labels)}\n")
        f.write(f"`define SNN_NUM_SYN        {len(csr_dst)}\n")
        f.write(f"`define SNN_TOPK_W1        {TOPK_W1}\n")
        f.write(f"`define SNN_FINETUNE_EPOCHS {FINETUNE_EPOCHS}\n")
        f.write(f"`define SNN_BIAS_MODE      \"{BIAS_MODE}\"\n")
        f.write(f"`define SNN_READOUT        \"{READOUT}\"\n")
        f.write(f"`define SNN_PRUNE_MODE     \"{PRUNE_MODE}\"\n")
        f.write(f"`define SNN_MIN_W1_PER_HIDDEN {MIN_W1_PER_HIDDEN}\n")
        f.write(f"`define SNN_ACTIVITY_LAMBDA {ACTIVITY_LAMBDA}\n")
        f.write(f"`define SNN_MEMBRANE_WIDTH {MEMBRANE_WIDTH}\n")
        f.write(f"`define SNN_HW_WRAP        {1 if HW_WRAP else 0}\n")
        f.write(f"`define SNN_ANN_HIDDEN_SHIFT {hidden_shift}\n")
        f.write(f"`define SNN_ANN_NUM_MACS   {N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT}\n")
        f.write("`endif\n")

    return len(csr_dst)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_ds, test_ds = load_tonic_nmnist()

    print("=" * 64)
    print("  N-MNIST sparse trained benchmark export")
    print(f"  network: {N_INPUT} -> {N_HIDDEN} -> {N_OUTPUT}")
    print(f"  timesteps/sample: {T_STEPS}")
    print(f"  bias mode: {BIAS_MODE}")
    print(f"  readout: {READOUT}")
    print(f"  prune mode: {PRUNE_MODE}")
    print(f"  membrane width: {MEMBRANE_WIDTH}  hardware wrap model: {int(HW_WRAP)}")
    if ACTIVITY_LAMBDA > 0.0:
        print(f"  sparse activity penalty: {ACTIVITY_LAMBDA:g}")
    train_indices = select_indices(train_ds, NUM_TRAIN, "train")
    test_indices = select_indices(test_ds, NUM_TEST_RTL, "test")
    print(f"  train samples used: {len(train_indices)}")
    print(f"  test samples exported: {len(test_indices)}")
    print("=" * 64)

    train_spikes, train_labels = load_split(train_ds, train_indices)
    test_spikes, test_labels = load_split(test_ds, test_indices)
    train_counts = train_spikes.sum(axis=1).astype(np.int32)
    test_counts = test_spikes.sum(axis=1).astype(np.int32)

    model, train_acc, test_acc = train_mlp(
        train_counts, train_labels, test_counts, test_labels)
    w1_dense = model[0].weight.detach().cpu().numpy()
    w2_dense = model[2].weight.detach().cpu().numpy()
    mask_np = topk_mask(w1_dense, train_counts, w2_dense)
    with torch.no_grad():
        model[0].weight.mul_(torch.tensor(mask_np, dtype=torch.float32))
    model, sparse_train_acc, sparse_test_acc = finetune_sparse(
        model, mask_np, train_counts, train_labels, test_counts, test_labels)
    w1_sparse = model[0].weight.detach().cpu().numpy()
    w2 = model[2].weight.detach().cpu().numpy()
    b1 = model[0].bias.detach().cpu().numpy() if model[0].bias is not None else None
    b2 = model[2].bias.detach().cpu().numpy() if model[2].bias is not None else None
    w1q, w2q, b1_spike, b2_spike, b1_ann, b2_ann = quantize_folded(w1_sparse, w2, b1, b2)

    hidden_shift = choose_ann_hidden_shift(train_counts, train_labels, w1q, w2q, b1_ann, b2_ann)
    ann_acc = simulate_ann_int(test_counts, test_labels, w1q, w2q, b1_ann, b2_ann, hidden_shift)

    tune_n = min(TUNE_SUBSET, len(train_labels))
    eval_n = min(EVAL_SUBSET, len(test_labels))
    print("  precomputing SNN hidden inputs")
    tune_hin = precompute_hidden_input(train_spikes[:tune_n], w1q, b1_spike)
    eval_hin = precompute_hidden_input(test_spikes[:eval_n], w1q, b1_spike)
    tune_input_deliveries = estimate_input_deliveries(
        train_spikes[:tune_n], w1q, b1_spike, b2_spike)
    best_score, best_acc, best_ops, best_thr = -1.0, -1.0, 0, None
    for thr in THR_RANGE:
        acc, stats = simulate_snn_hin(
            tune_hin, train_labels[:tune_n], w2q, b2_spike, thr, return_stats=True)
        total_deliveries = tune_input_deliveries + int(stats["hidden_deliveries"])
        score = threshold_score(acc, total_deliveries, tune_n)
        if score > best_score or (score == best_score and acc > best_acc):
            best_score, best_acc, best_ops, best_thr = score, acc, total_deliveries, thr
    snn_acc, snn_stats = simulate_snn_hin(
        eval_hin, test_labels[:eval_n], w2q, b2_spike, best_thr, return_stats=True)
    eval_input_deliveries = estimate_input_deliveries(
        test_spikes[:eval_n], w1q, b1_spike, b2_spike)
    eval_deliveries = eval_input_deliveries + int(snn_stats["hidden_deliveries"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.abspath(os.path.join(script_dir, "..", "sim"))
    n_syn = export(sim_dir, test_spikes, test_labels, test_counts,
                   w1q, w2q, b1_spike, b2_spike, b1_ann, b2_ann,
                   best_thr, hidden_shift)

    avg_events = float(test_spikes.sum()) / max(len(test_labels), 1)
    metrics = {
        "dataset": "nmnist",
        "selection": SELECTION,
        "label_hist": [int(v) for v in np.bincount(test_labels, minlength=N_OUTPUT)],
        "n_input": N_INPUT,
        "n_hidden": N_HIDDEN,
        "n_output": N_OUTPUT,
        "t_steps": T_STEPS,
        "num_train": int(len(train_labels)),
        "num_test": int(len(test_labels)),
        "epochs": EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
        "topk_w1": TOPK_W1,
        "bias_mode": BIAS_MODE,
        "readout": READOUT,
        "prune_mode": PRUNE_MODE,
        "min_w1_per_hidden": MIN_W1_PER_HIDDEN,
        "activity_lambda": ACTIVITY_LAMBDA,
        "membrane_width": MEMBRANE_WIDTH,
        "hw_wrap": int(HW_WRAP),
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "threshold": int(best_thr),
        "threshold_tune_acc_pct": best_acc * 100.0,
        "threshold_tune_deliveries": int(best_ops),
        "float_train_acc_pct": train_acc * 100.0,
        "float_test_acc_pct": test_acc * 100.0,
        "sparse_float_train_acc_pct": sparse_train_acc * 100.0,
        "sparse_float_test_acc_pct": sparse_test_acc * 100.0,
        "ann_int_acc_pct": ann_acc * 100.0,
        "ann_hidden_shift": int(hidden_shift),
        "snn_eval_acc_pct": snn_acc * 100.0,
        "snn_eval_deliveries": int(eval_deliveries),
        "snn_eval_deliveries_per_sample": float(eval_deliveries / max(eval_n, 1)),
        "snn_eval_hidden_spikes": int(snn_stats["hidden_spikes"]),
        "snn_eval_output_spikes": int(snn_stats["output_spikes"]),
        "csr_synapses": int(n_syn),
        "bias_hidden_synapses": int(np.count_nonzero(b1_spike)),
        "bias_output_synapses": int(np.count_nonzero(b2_spike)),
        "avg_binned_input_events_per_sample": avg_events,
        "dense_ann_macs_per_sample": N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT,
        "dense_to_sparse_storage_ratio": (
            (N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT) / max(n_syn, 1)
        ),
    }
    metrics["accuracy_per_ksynapse"] = metrics["snn_eval_acc_pct"] / max(n_syn / 1000.0, 1e-9)
    metrics["accuracy_per_kdelivery"] = (
        metrics["snn_eval_acc_pct"] / max(metrics["snn_eval_deliveries_per_sample"] / 1000.0, 1e-9)
    )
    write_export_metrics(sim_dir, metrics)

    print("=" * 64)
    print("  N-MNIST export complete")
    print(f"  float count MLP train/test accuracy: {train_acc * 100:.2f}% / {test_acc * 100:.2f}%")
    print(f"  sparse float MLP train/test accuracy: {sparse_train_acc * 100:.2f}% / {sparse_test_acc * 100:.2f}%")
    print(f"  sparse W1 top-k per hidden neuron: {TOPK_W1}")
    print(f"  prune mode: {PRUNE_MODE}  min W1/hidden: {MIN_W1_PER_HIDDEN}")
    print(f"  sparse finetune epochs: {FINETUNE_EPOCHS}")
    print(f"  bias mode: {BIAS_MODE}  hidden/output bias synapses: {np.count_nonzero(b1_spike)} / {np.count_nonzero(b2_spike)}")
    print(f"  readout: {READOUT}  membrane width: {MEMBRANE_WIDTH}  hw-wrap: {int(HW_WRAP)}")
    if ACTIVITY_LAMBDA > 0.0:
        print(f"  sparse activity penalty: {ACTIVITY_LAMBDA:g}")
    print(f"  sparse INT8 ANN exported-test accuracy: {ann_acc * 100:.2f}%")
    print(f"  sparse INT8 ANN hidden shift: {hidden_shift}")
    print(f"  sparse SNN threshold: {best_thr}  objective={THRESHOLD_OBJECTIVE}")
    print(f"  sparse SNN eval accuracy ({eval_n} samples): {snn_acc * 100:.2f}%")
    print(f"  SNN estimated deliveries/sample ({eval_n} samples): {eval_deliveries / max(eval_n, 1):.2f}")
    print(f"  CSR synapses: {n_syn}")
    print(f"  average binned input events/sample: {avg_events:.2f}")
    print(f"  dense ANN MACs/sample: {N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT}")
    print(f"  export metrics: {os.path.join(sim_dir, 'export_nmnist_metrics.json')}")
    print(f"  exported to: {sim_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
