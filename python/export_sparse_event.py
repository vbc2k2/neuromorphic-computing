"""
export_sparse_event.py - Generate a sparse event-native benchmark for RTL.

This is not a public dataset benchmark. It is a controlled microbenchmark that
answers one architectural question:

    When the input stream and weights are truly sparse, does the event-driven
    SNN RTL avoid work that the dense ANN RTL still performs?

The generated task is intentionally simple and exactly reproducible:
    - 10 classes.
    - 1024 possible input event addresses.
    - Each class owns a small pool of input addresses.
    - Each sample emits one class-coded event per timestep, plus occasional
      noise events from other classes.
    - CSR connectivity is sparse: each class input address connects to one
      hidden detector, and each hidden detector connects to its class output.
    - The ANN baseline receives a static event-count vector for the same sample
      and uses the same sparse weights stored in dense memories.

The event SNN should require far fewer operations than the dense ANN baseline.
Use this to validate the research direction before moving to N-MNIST/DVS.
"""

import os
import random
from typing import List, Tuple

import numpy as np


N_INPUT = 1024
N_HIDDEN = 64
N_OUTPUT = 10
T_STEPS = 32
NUM_TEST_RTL = 300

ID_BIAS = N_INPUT
ID_HIDDEN_BASE = N_INPUT + 1
ID_OUTPUT_BASE = N_INPUT + 1 + N_HIDDEN
N_TOTAL = N_INPUT + 1 + N_HIDDEN + N_OUTPUT

THRESHOLD = 100
LEAK = 1
PIXEL_MAX = 255
ANN_HIDDEN_SHIFT = 4
SEED = 7

CLASS_POOL = 32
EVENTS_PER_STEP = 1
NOISE_PROB = 0.10


def class_for_input(pixel: int) -> int:
    return pixel // CLASS_POOL if pixel < CLASS_POOL * N_OUTPUT else -1


def hidden_for_class(cls: int) -> int:
    return ID_HIDDEN_BASE + cls


def build_samples() -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(SEED)
    spikes = np.zeros((NUM_TEST_RTL, T_STEPS, N_INPUT), dtype=np.uint8)
    labels = np.zeros(NUM_TEST_RTL, dtype=np.int64)

    for img in range(NUM_TEST_RTL):
        cls = img % N_OUTPUT
        labels[img] = cls
        own_base = cls * CLASS_POOL
        for t in range(T_STEPS):
            for _ in range(EVENTS_PER_STEP):
                p = own_base + rng.randrange(CLASS_POOL)
                spikes[img, t, p] = 1
            if rng.random() < NOISE_PROB:
                noise_cls = rng.randrange(N_OUTPUT - 1)
                if noise_cls >= cls:
                    noise_cls += 1
                p = noise_cls * CLASS_POOL + rng.randrange(CLASS_POOL)
                spikes[img, t, p] = 1

    return spikes, labels


def build_sparse_weights():
    w1 = np.zeros((N_HIDDEN, N_INPUT), dtype=np.int8)
    b1 = np.zeros(N_HIDDEN, dtype=np.int32)
    w2 = np.zeros((N_OUTPUT, N_HIDDEN), dtype=np.int8)
    b2 = np.zeros(N_OUTPUT, dtype=np.int32)

    for cls in range(N_OUTPUT):
        h = cls
        for p in range(cls * CLASS_POOL, (cls + 1) * CLASS_POOL):
            w1[h, p] = 127
        w2[cls, h] = 127

    return w1, b1, w2, b2


def build_sparse_csr(w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray):
    csr_dst: List[int] = []
    csr_weight: List[int] = []
    src_start = [0] * N_TOTAL
    src_count = [0] * N_TOTAL

    for src in range(N_TOTAL):
        src_start[src] = len(csr_dst)
        targets = []
        if src < N_INPUT:
            cls = class_for_input(src)
            if cls >= 0:
                targets.append((hidden_for_class(cls), int(w1[cls, src])))
        elif ID_HIDDEN_BASE <= src < ID_HIDDEN_BASE + N_HIDDEN:
            h = src - ID_HIDDEN_BASE
            for out in range(N_OUTPUT):
                weight = int(w2[out, h])
                if weight != 0:
                    targets.append((ID_OUTPUT_BASE + out, weight))
        # Bias intentionally has no fanout in this sparse benchmark.
        for dst, weight in targets:
            csr_dst.append(dst)
            csr_weight.append(weight)
        src_count[src] = len(targets)

    return csr_dst, csr_weight, src_start, src_count


def simulate_snn(spikes: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for img in range(spikes.shape[0]):
        hidden_counts = np.zeros(N_OUTPUT, dtype=np.int32)
        output_counts = np.zeros(N_OUTPUT, dtype=np.int32)
        prev_hidden = np.zeros(N_OUTPUT, dtype=np.uint8)
        for t in range(T_STEPS):
            output_counts += prev_hidden
            current_hidden = np.zeros(N_OUTPUT, dtype=np.uint8)
            for p in np.flatnonzero(spikes[img, t]):
                cls = class_for_input(int(p))
                if cls >= 0:
                    current_hidden[cls] = 1
                    hidden_counts[cls] += 1
            prev_hidden = current_hidden
        pred = int(np.argmax(output_counts))
        correct += int(pred == labels[img])
    return correct / max(spikes.shape[0], 1)


def simulate_ann_counts(spikes: np.ndarray, labels: np.ndarray) -> float:
    w1, b1, w2, b2 = build_sparse_weights()
    counts = spikes.sum(axis=1).astype(np.int32)
    h_raw = np.maximum(0, counts @ w1.T.astype(np.int32) + b1)
    h = np.minimum(h_raw >> ANN_HIDDEN_SHIFT, 255).astype(np.int32)
    y = h @ w2.T.astype(np.int32) + b2
    pred = np.argmax(y, axis=1)
    return float((pred == labels).mean())


def write_memories(sim_dir: str, spikes: np.ndarray, labels: np.ndarray):
    os.makedirs(sim_dir, exist_ok=True)
    w1, b1, w2, b2 = build_sparse_weights()
    csr_dst, csr_weight, src_start, src_count = build_sparse_csr(w1, b1, w2, b2)

    with open(os.path.join(sim_dir, "csr_dst.mem"), "w") as f:
        for d in csr_dst:
            f.write(f"{d & 0xFFFF:04X}\n")
    with open(os.path.join(sim_dir, "csr_weight.mem"), "w") as f:
        for w in csr_weight:
            f.write(f"{w & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, "csr_start.mem"), "w") as f:
        for v in src_start:
            f.write(f"{v & 0xFFFFFF:06X}\n")
    with open(os.path.join(sim_dir, "csr_count.mem"), "w") as f:
        for v in src_count:
            f.write(f"{v & 0xFFFFFF:06X}\n")

    with open(os.path.join(sim_dir, "snn_spikes.mem"), "w") as f:
        for img in spikes:
            for t in range(T_STEPS):
                f.write("".join("1" if img[t, p] else "0"
                                for p in range(N_INPUT - 1, -1, -1)) + "\n")

    with open(os.path.join(sim_dir, "snn_labels.mem"), "w") as f:
        for lab in labels:
            f.write(f"{int(lab)}\n")

    pixels = np.minimum(spikes.sum(axis=1), PIXEL_MAX).astype(np.uint8)
    with open(os.path.join(sim_dir, "ann_pixels.mem"), "w") as f:
        for img in pixels:
            for pix in img:
                f.write(f"{int(pix) & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, "ann_w1.mem"), "w") as f:
        for row in w1:
            for w in row:
                f.write(f"{int(w) & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, "ann_b1.mem"), "w") as f:
        for b in b1:
            f.write(f"{int(b) & 0xFFFFFFFF:08X}\n")
    with open(os.path.join(sim_dir, "ann_w2.mem"), "w") as f:
        for row in w2:
            for w in row:
                f.write(f"{int(w) & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, "ann_b2.mem"), "w") as f:
        for b in b2:
            f.write(f"{int(b) & 0xFFFFFFFF:08X}\n")

    with open(os.path.join(sim_dir, "snn_config.vh"), "w") as f:
        f.write("// Auto-generated by python/export_sparse_event.py - do not edit by hand\n")
        f.write("`ifndef SNN_CONFIG_VH\n`define SNN_CONFIG_VH\n")
        f.write("`define SNN_DATASET        \"sparse_event\"\n")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.abspath(os.path.join(script_dir, "..", "sim"))
    spikes, labels = build_samples()
    n_syn = write_memories(sim_dir, spikes, labels)
    snn_acc = simulate_snn(spikes, labels)
    ann_acc = simulate_ann_counts(spikes, labels)
    ann_macs = N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT
    avg_events = float(spikes.sum()) / len(labels)

    print("=" * 64)
    print("  Sparse event-native benchmark export")
    print(f"  network: {N_INPUT} -> {N_HIDDEN} -> {N_OUTPUT}  ({N_TOTAL} neurons)")
    print(f"  timesteps/image: {T_STEPS}")
    print(f"  test images: {len(labels)}")
    print(f"  CSR synapses: {n_syn}")
    print(f"  average input events/image: {avg_events:.2f}")
    print(f"  golden SNN accuracy: {snn_acc * 100:.2f}%")
    print(f"  dense ANN-count accuracy: {ann_acc * 100:.2f}%")
    print(f"  dense ANN MACs/image: {ann_macs}")
    print(f"  exported to: {sim_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
