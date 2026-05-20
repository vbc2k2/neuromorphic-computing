"""
export_shd.py - Export the public SHD spike-audio dataset to the RTL format.

SHD (Spiking Heidelberg Digits) is a public spike-based spoken digit dataset.
It has 700 input channels and 20 classes. This script uses a deliberately small
sparse prototype classifier so we can test the hardware question first:

    Does event-driven RTL avoid work on a real sparse spike dataset?

The model is simple:
    - Bin each SHD event stream into T_STEPS binary timesteps.
    - Learn class prototypes by selecting channels that are more active for
      each class than for other classes.
    - Map selected input channels to one hidden detector per class.
    - Map each class detector to the corresponding output.

This is a baseline/export path, not a final ML model. If the hardware result is
promising, the next step is a trained sparse SNN/ANN model with proper train/test
accuracy reporting.

Prerequisite:
    python3 -m pip install --user tonic h5py

Usage:
    python3 python/export_shd.py
    bash tools/run_verilator_event.sh 0 299 _shd
    bash tools/run_verilator_ann.sh 0 299 _shd
"""

import os
from typing import Iterable, List, Tuple

import numpy as np


N_INPUT = 700
N_OUTPUT = 20
N_HIDDEN = N_OUTPUT
N_TOTAL = N_INPUT + 1 + N_HIDDEN + N_OUTPUT

ID_BIAS = N_INPUT
ID_HIDDEN_BASE = N_INPUT + 1
ID_OUTPUT_BASE = N_INPUT + 1 + N_HIDDEN

T_STEPS = int(os.environ.get("SHD_T_STEPS", "100"))
NUM_TRAIN = int(os.environ.get("SHD_NUM_TRAIN", "2000"))
NUM_TEST_RTL = int(os.environ.get("SHD_NUM_TEST_RTL", "300"))
FEATURES_PER_CLASS = int(os.environ.get("SHD_FEATURES_PER_CLASS", "32"))

THRESHOLD = 1
LEAK = 1
ANN_HIDDEN_SHIFT = 0
SEED = 11


def load_tonic_shd():
    try:
        import tonic
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'tonic'. Install it in your active environment:\n"
            "  python3 -m pip install --user tonic h5py\n"
            "or, in conda:\n"
            "  python3 -m pip install tonic h5py"
        ) from exc

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.environ.get("SNN_DATA_DIR", os.path.join(repo_root, "data"))
    train = tonic.datasets.SHD(save_to=data_dir, train=True)
    test = tonic.datasets.SHD(save_to=data_dir, train=False)
    return train, test


def event_fields(events) -> Tuple[np.ndarray, np.ndarray]:
    names = events.dtype.names
    if names is None:
        arr = np.asarray(events)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Unsupported SHD event array shape")
        # Tonic's audio examples use txp ordering for audio datasets.
        return arr[:, 0], arr[:, 1]

    if "t" not in names:
        raise ValueError(f"SHD event dtype has no time field: {names}")

    if "x" in names:
        channel = events["x"]
    elif "p" in names:
        channel = events["p"]
    else:
        raise ValueError(f"SHD event dtype has no channel field: {names}")
    return events["t"], channel


def events_to_spikes(events) -> np.ndarray:
    spikes = np.zeros((T_STEPS, N_INPUT), dtype=np.uint8)
    if len(events) == 0:
        return spikes

    t, channel = event_fields(events)
    t = np.asarray(t, dtype=np.float64)
    channel = np.asarray(channel, dtype=np.int64)

    valid = (channel >= 0) & (channel < N_INPUT)
    if not np.any(valid):
        return spikes
    t = t[valid]
    channel = channel[valid]

    t0 = float(t.min())
    t1 = float(t.max())
    if t1 <= t0:
        bins = np.zeros_like(channel)
    else:
        bins = ((t - t0) * T_STEPS / (t1 - t0 + 1e-12)).astype(np.int64)
        bins = np.clip(bins, 0, T_STEPS - 1)
    spikes[bins, channel] = 1
    return spikes


def load_split(dataset, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    n = min(limit, len(dataset))
    spikes = np.zeros((n, T_STEPS, N_INPUT), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        events, label = dataset[i]
        spikes[i] = events_to_spikes(events)
        labels[i] = int(label)
        if (i + 1) % 250 == 0:
            print(f"  loaded {i + 1}/{n} samples")
    return spikes, labels


def select_features(train_spikes: np.ndarray, train_labels: np.ndarray) -> List[List[int]]:
    counts = train_spikes.sum(axis=1).astype(np.float32)
    global_mean = counts.mean(axis=0)
    selected: List[List[int]] = []
    for cls in range(N_OUTPUT):
        mask = train_labels == cls
        if not np.any(mask):
            selected.append([])
            continue
        class_mean = counts[mask].mean(axis=0)
        score = class_mean - global_mean
        order = np.argsort(score)[::-1]
        feats = [int(ch) for ch in order[:FEATURES_PER_CLASS] if score[ch] > 0]
        if len(feats) < FEATURES_PER_CLASS:
            feats = [int(ch) for ch in order[:FEATURES_PER_CLASS]]
        selected.append(feats)
    return selected


def predict_from_features(spikes: np.ndarray, selected: List[List[int]]) -> np.ndarray:
    counts = spikes.sum(axis=1)
    scores = np.zeros((spikes.shape[0], N_OUTPUT), dtype=np.int32)
    for cls, feats in enumerate(selected):
        if feats:
            scores[:, cls] = counts[:, feats].sum(axis=1)
    return np.argmax(scores, axis=1)


def build_memories(selected: List[List[int]]):
    w1 = np.zeros((N_HIDDEN, N_INPUT), dtype=np.int8)
    b1 = np.zeros(N_HIDDEN, dtype=np.int32)
    w2 = np.zeros((N_OUTPUT, N_HIDDEN), dtype=np.int8)
    b2 = np.zeros(N_OUTPUT, dtype=np.int32)

    for cls, feats in enumerate(selected):
        for ch in feats:
            w1[cls, ch] = 1
        w2[cls, cls] = 1

    csr_dst: List[int] = []
    csr_weight: List[int] = []
    src_start = [0] * N_TOTAL
    src_count = [0] * N_TOTAL

    by_src: List[List[Tuple[int, int]]] = [[] for _ in range(N_TOTAL)]
    for cls, feats in enumerate(selected):
        hid = ID_HIDDEN_BASE + cls
        out = ID_OUTPUT_BASE + cls
        for ch in feats:
            by_src[ch].append((hid, 1))
        by_src[hid].append((out, 1))

    for src in range(N_TOTAL):
        src_start[src] = len(csr_dst)
        for dst, weight in by_src[src]:
            csr_dst.append(dst)
            csr_weight.append(weight)
        src_count[src] = len(by_src[src])

    return w1, b1, w2, b2, csr_dst, csr_weight, src_start, src_count


def write_hex(path: str, values: Iterable[int], width: int):
    mask = (1 << (4 * width)) - 1
    with open(path, "w") as f:
        for value in values:
            f.write(f"{int(value) & mask:0{width}X}\n")


def export(sim_dir: str, spikes: np.ndarray, labels: np.ndarray,
           selected: List[List[int]]):
    os.makedirs(sim_dir, exist_ok=True)
    w1, b1, w2, b2, csr_dst, csr_weight, src_start, src_count = build_memories(selected)

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

    pixels = np.minimum(spikes.sum(axis=1), 255).astype(np.uint8)
    write_hex(os.path.join(sim_dir, "ann_pixels.mem"), pixels.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_w1.mem"), w1.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_b1.mem"), b1, 8)
    write_hex(os.path.join(sim_dir, "ann_w2.mem"), w2.flatten(), 2)
    write_hex(os.path.join(sim_dir, "ann_b2.mem"), b2, 8)

    with open(os.path.join(sim_dir, "snn_config.vh"), "w") as f:
        f.write("// Auto-generated by python/export_shd.py - do not edit by hand\n")
        f.write("`ifndef SNN_CONFIG_VH\n`define SNN_CONFIG_VH\n")
        f.write("`define SNN_DATASET        \"shd\"\n")
        f.write(f"`define SNN_N_TOTAL        {N_TOTAL}\n")
        f.write(f"`define SNN_N_INPUT        {N_INPUT}\n")
        f.write(f"`define SNN_N_HIDDEN       {N_HIDDEN}\n")
        f.write(f"`define SNN_N_OUTPUT       {N_OUTPUT}\n")
        f.write(f"`define SNN_ID_BIAS        {ID_BIAS}\n")
        f.write(f"`define SNN_ID_HIDDEN_BASE {ID_HIDDEN_BASE}\n")
        f.write(f"`define SNN_ID_OUTPUT_BASE {ID_OUTPUT_BASE}\n")
        f.write(f"`define SNN_T_STEPS        {T_STEPS}\n")
        f.write(f"`define SNN_THRESHOLD      {THRESHOLD}\n")
        f.write(f"`define SNN_LEAK           {LEAK}\n")
        f.write(f"`define SNN_NUM_TEST       {len(labels)}\n")
        f.write(f"`define SNN_NUM_SYN        {len(csr_dst)}\n")
        f.write(f"`define SNN_ANN_HIDDEN_SHIFT {ANN_HIDDEN_SHIFT}\n")
        f.write(f"`define SNN_ANN_NUM_MACS   {N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT}\n")
        f.write("`endif\n")

    return len(csr_dst)


def main():
    np.random.seed(SEED)
    train_ds, test_ds = load_tonic_shd()

    print("=" * 64)
    print("  SHD public spike-audio benchmark export")
    print(f"  timesteps/sample: {T_STEPS}")
    print(f"  train samples used: {min(NUM_TRAIN, len(train_ds))}")
    print(f"  test samples exported: {min(NUM_TEST_RTL, len(test_ds))}")
    print("=" * 64)

    train_spikes, train_labels = load_split(train_ds, NUM_TRAIN)
    test_spikes, test_labels = load_split(test_ds, NUM_TEST_RTL)
    selected = select_features(train_spikes, train_labels)

    train_pred = predict_from_features(train_spikes, selected)
    test_pred = predict_from_features(test_spikes, selected)
    train_acc = float((train_pred == train_labels).mean())
    test_acc = float((test_pred == test_labels).mean())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.abspath(os.path.join(script_dir, "..", "sim"))
    n_syn = export(sim_dir, test_spikes, test_labels, selected)

    avg_events = float(test_spikes.sum()) / max(len(test_labels), 1)
    print("=" * 64)
    print("  SHD sparse prototype export complete")
    print(f"  network: {N_INPUT} -> {N_HIDDEN} -> {N_OUTPUT}  ({N_TOTAL} neurons)")
    print(f"  CSR synapses: {n_syn}")
    print(f"  average binned input events/sample: {avg_events:.2f}")
    print(f"  prototype train accuracy: {train_acc * 100:.2f}%")
    print(f"  prototype exported-test accuracy: {test_acc * 100:.2f}%")
    print(f"  dense ANN MACs/sample: {N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT}")
    print(f"  exported to: {sim_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
