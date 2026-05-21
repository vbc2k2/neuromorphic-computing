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

THR_RANGE = range(20, 1500, 20)
TUNE_SUBSET = int(os.environ.get("NMNIST_TUNE_SUBSET", "500"))
EVAL_SUBSET = int(os.environ.get("NMNIST_EVAL_SUBSET", "500"))

LEAK = 1
SEED = 13
SELECTION = os.environ.get("NMNIST_SELECTION", "balanced")


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
    model = nn.Sequential(
        nn.Linear(N_INPUT, N_HIDDEN, bias=False),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_OUTPUT, bias=False),
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


def topk_mask(w1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(w1, dtype=np.float32)
    k = min(TOPK_W1, w1.shape[1])
    for h in range(w1.shape[0]):
        idx = np.argsort(np.abs(w1[h]))[-k:]
        mask[h, idx] = 1.0
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
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            model[0].weight.grad.mul_(mask)
            opt.step()
            with torch.no_grad():
                model[0].weight.mul_(mask)
        if epoch in {0, FINETUNE_EPOCHS - 1}:
            acc = eval_model(model, test_counts, test_labels)
            print(f"  sparse finetune {epoch + 1:02d}/{FINETUNE_EPOCHS}: test-count acc {acc * 100:.2f}%")

    return model, eval_model(model, train_counts, train_labels), eval_model(model, test_counts, test_labels)


def quantize_signed(weights: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(weights)))
    if max_abs == 0:
        return np.zeros_like(weights, dtype=np.int8)
    q = np.round(weights * (127.0 / max_abs))
    return np.clip(q, -127, 127).astype(np.int8)


def quantize_folded(w1_sparse: np.ndarray, w2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-hidden W1 quantization, with the hidden scale folded into W2.

    The RTL has no floating scale multipliers. For hidden neuron h, qW1[h] is
    scaled independently for better int8 resolution. Dividing W2[:, h] by the
    same scale keeps the output logits equivalent up to one global W2 scale.
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
    w2q = quantize_signed(w2_folded)
    return w1q, w2q


def simulate_ann_int(counts: np.ndarray, labels: np.ndarray, w1q: np.ndarray,
                     w2q: np.ndarray, hidden_shift: int) -> float:
    h_raw = counts.astype(np.int32) @ w1q.T.astype(np.int32)
    h = np.maximum(h_raw, 0) >> hidden_shift
    h = np.minimum(h, 255).astype(np.int32)
    y = h @ w2q.T.astype(np.int32)
    pred = np.argmax(y, axis=1)
    return float((pred == labels).mean())


def choose_ann_hidden_shift(counts: np.ndarray, labels: np.ndarray,
                            w1q: np.ndarray, w2q: np.ndarray) -> int:
    best_shift, best_acc = 0, -1.0
    for shift in range(16):
        acc = simulate_ann_int(counts, labels, w1q, w2q, shift)
        if acc > best_acc:
            best_shift, best_acc = shift, acc
    return best_shift


def lif_leak(mem: np.ndarray) -> np.ndarray:
    return np.where(mem > LEAK, mem - LEAK, np.where(mem > 0, 0, mem))


def precompute_hidden_input(spikes: np.ndarray, w1q: np.ndarray) -> np.ndarray:
    w1i = w1q.astype(np.int32)
    hin = np.zeros((spikes.shape[0], T_STEPS, N_HIDDEN), dtype=np.int32)
    for t in range(T_STEPS):
        hin[:, t, :] = spikes[:, t, :].astype(np.int32) @ w1i.T
    return hin


def simulate_snn_hin(hin: np.ndarray, labels: np.ndarray,
                     w2q: np.ndarray, threshold: int) -> float:
    num = hin.shape[0]
    mem_h = np.zeros((num, N_HIDDEN), dtype=np.int32)
    mem_o = np.zeros((num, N_OUTPUT), dtype=np.int32)
    hidden_prev = np.zeros((num, N_HIDDEN), dtype=np.int32)
    out_count = np.zeros((num, N_OUTPUT), dtype=np.int32)
    w2i = w2q.astype(np.int32)

    for t in range(T_STEPS):
        mem_h = lif_leak(mem_h) + hin[:, t, :]
        hidden = mem_h >= threshold
        mem_h[hidden] = 0

        oin = hidden_prev @ w2i.T
        mem_o = lif_leak(mem_o) + oin
        out = mem_o >= threshold
        mem_o[out] = 0
        out_count += out.astype(np.int32)
        hidden_prev = hidden.astype(np.int32)

    pred = np.argmax(out_count, axis=1)
    return float((pred == labels).mean())


def build_csr(w1q: np.ndarray, w2q: np.ndarray):
    by_src = [[] for _ in range(N_TOTAL)]
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


def export(sim_dir: str, spikes: np.ndarray, labels: np.ndarray,
           counts: np.ndarray, w1q: np.ndarray, w2q: np.ndarray,
           threshold: int, hidden_shift: int):
    os.makedirs(sim_dir, exist_ok=True)
    csr_dst, csr_weight, src_start, src_count = build_csr(w1q, w2q)

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
    write_hex(os.path.join(sim_dir, "ann_b1.mem"), np.zeros(N_HIDDEN, dtype=np.int32), 8)
    write_hex(os.path.join(sim_dir, "ann_w2.mem"), w2q.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_b2.mem"), np.zeros(N_OUTPUT, dtype=np.int32), 8)

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
    mask_np = topk_mask(w1_dense)
    with torch.no_grad():
        model[0].weight.mul_(torch.tensor(mask_np, dtype=torch.float32))
    model, sparse_train_acc, sparse_test_acc = finetune_sparse(
        model, mask_np, train_counts, train_labels, test_counts, test_labels)
    w1_sparse = model[0].weight.detach().cpu().numpy()
    w2 = model[2].weight.detach().cpu().numpy()
    w1q, w2q = quantize_folded(w1_sparse, w2)

    hidden_shift = choose_ann_hidden_shift(train_counts, train_labels, w1q, w2q)
    ann_acc = simulate_ann_int(test_counts, test_labels, w1q, w2q, hidden_shift)

    tune_n = min(TUNE_SUBSET, len(train_labels))
    eval_n = min(EVAL_SUBSET, len(test_labels))
    print("  precomputing SNN hidden inputs")
    tune_hin = precompute_hidden_input(train_spikes[:tune_n], w1q)
    eval_hin = precompute_hidden_input(test_spikes[:eval_n], w1q)
    best_acc, best_thr = -1.0, None
    for thr in THR_RANGE:
        acc = simulate_snn_hin(tune_hin, train_labels[:tune_n], w2q, thr)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    snn_acc = simulate_snn_hin(eval_hin, test_labels[:eval_n], w2q, best_thr)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.abspath(os.path.join(script_dir, "..", "sim"))
    n_syn = export(sim_dir, test_spikes, test_labels, test_counts,
                   w1q, w2q, best_thr, hidden_shift)

    avg_events = float(test_spikes.sum()) / max(len(test_labels), 1)
    print("=" * 64)
    print("  N-MNIST export complete")
    print(f"  float count MLP train/test accuracy: {train_acc * 100:.2f}% / {test_acc * 100:.2f}%")
    print(f"  sparse float MLP train/test accuracy: {sparse_train_acc * 100:.2f}% / {sparse_test_acc * 100:.2f}%")
    print(f"  sparse W1 top-k per hidden neuron: {TOPK_W1}")
    print(f"  sparse finetune epochs: {FINETUNE_EPOCHS}")
    print(f"  sparse INT8 ANN exported-test accuracy: {ann_acc * 100:.2f}%")
    print(f"  sparse INT8 ANN hidden shift: {hidden_shift}")
    print(f"  sparse SNN threshold: {best_thr}")
    print(f"  sparse SNN eval accuracy ({eval_n} samples): {snn_acc * 100:.2f}%")
    print(f"  CSR synapses: {n_syn}")
    print(f"  average binned input events/sample: {avg_events:.2f}")
    print(f"  dense ANN MACs/sample: {N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT}")
    print(f"  exported to: {sim_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
