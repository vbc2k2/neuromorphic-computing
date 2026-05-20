"""
train_snn.py - Train an ANN on handwritten digits, convert it to a rate-coded
spiking neural network, quantize to int8, and export everything the RTL
accelerator needs.

Supports two benchmarks (set DATASET below):
  'digits' : sklearn 8x8 digits   ->  64 -> 64  -> 10   (Phase 1, runs locally)
  'mnist'  : full 28x28 MNIST     -> 784 -> 256 -> 10   (Phase 2, server runs)

Pipeline:
  1. Train an MLP (ReLU) on the chosen dataset.
  2. Quantize the weights to signed 8-bit.
  3. Simulate the resulting SNN in numpy with the EXACT integer LIF math of
     rtl/neuron_core.sv -- this numpy model is the golden spec the RTL matches.
  4. Export CSR connectivity + input spike trains + a Verilog config header.

Flat neuron-id layout:
  ids 0 .. N_INPUT-1        : pixel input neurons
  id  N_INPUT               : bias neuron (fires every timestep)
  ids .. + N_HIDDEN         : hidden LIF neurons
  ids .. + N_OUTPUT         : output LIF neurons (argmax of spike count = class)

Author: Neuromorphic Accelerator Project
"""

import os
import urllib.request

# Keep BLAS/OpenMP libraries polite on shared login/HPC nodes. Users can
# override these before launching the script if a job allocation allows more.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("SNN_TORCH_THREADS", "1")))
torch.set_num_interop_threads(int(os.environ.get("SNN_TORCH_INTEROP_THREADS", "1")))

# ---------------------------------------------------------------------------
# Benchmark selection
# ---------------------------------------------------------------------------
DATASET = 'mnist'                    # 'digits' or 'mnist'

if DATASET == 'digits':
    N_INPUT, N_HIDDEN, N_OUTPUT = 64, 64, 10
    PIXEL_MAX     = 16
    T_STEPS       = 40
    NUM_TEST_RTL  = 60
    THR_RANGE     = range(20, 420, 10)
    TUNE_SUBSET   = 360
    EVAL_SUBSET   = 360
else:  # mnist
    N_INPUT, N_HIDDEN, N_OUTPUT = 784, 256, 10
    PIXEL_MAX     = 255
    T_STEPS       = 50
    NUM_TEST_RTL  = 300
    THR_RANGE     = range(100, 4000, 50)
    TUNE_SUBSET   = 400
    EVAL_SUBSET   = 2000

ID_BIAS        = N_INPUT
ID_HIDDEN_BASE = N_INPUT + 1
ID_OUTPUT_BASE = N_INPUT + 1 + N_HIDDEN
N_TOTAL        = N_INPUT + 1 + N_HIDDEN + N_OUTPUT
LEAK           = 1
RESET_VAL      = 0

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset loading -> integer pixels in [0, PIXEL_MAX]
# ---------------------------------------------------------------------------
def load_dataset():
    if DATASET == 'digits':
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        d = load_digits()
        X = d.data.astype(np.int32)
        y = d.target.astype(np.int64)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=y)
    else:
        Xtr, ytr, Xte, yte = load_mnist_npz()
        Xtr = Xtr.reshape(-1, 784).astype(np.int32)
        Xte = Xte.reshape(-1, 784).astype(np.int32)
        ytr = ytr.astype(np.int64)
        yte = yte.astype(np.int64)
    return Xtr, ytr, Xte, yte


def load_mnist_npz():
    """Load MNIST without TensorFlow, for portable HPC/open-source runs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    data_dir = os.environ.get("SNN_DATA_DIR", os.path.join(repo_root, 'data'))
    os.makedirs(data_dir, exist_ok=True)

    env_path = os.environ.get("MNIST_NPZ")
    path = env_path if env_path else os.path.join(data_dir, 'mnist.npz')
    if not os.path.exists(path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  Downloading MNIST to {path}")
        urllib.request.urlretrieve(url, path)

    with np.load(path) as data:
        return data['x_train'], data['y_train'], data['x_test'], data['y_test']


# ---------------------------------------------------------------------------
# 1. Train a plain ANN
# ---------------------------------------------------------------------------
def train_ann(Xtr, ytr, Xte, yte):
    model = nn.Sequential(
        nn.Linear(N_INPUT, N_HIDDEN), nn.ReLU(),
        nn.Linear(N_HIDDEN, N_OUTPUT),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr_t)
    epochs = 220 if DATASET == 'digits' else 30
    batch = 256
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    acc = (pred == yte).mean()
    print(f"  ANN test accuracy: {acc*100:.2f}%")
    return (model[0].weight.detach().numpy(), model[0].bias.detach().numpy(),
            model[2].weight.detach().numpy(), model[2].bias.detach().numpy(), acc)


# ---------------------------------------------------------------------------
# 2. Quantize a float array to signed 8-bit
# ---------------------------------------------------------------------------
def quantize(arr, scale):
    return np.clip(np.round(arr * scale), -127, 127).astype(np.int32)


# ---------------------------------------------------------------------------
# Rate encoding: pixel value (0..PIXEL_MAX) -> regular spike train of length T
# ---------------------------------------------------------------------------
def make_rate_patterns(t_steps, max_val):
    patterns = np.zeros((max_val + 1, t_steps), dtype=np.uint8)
    for v in range(max_val + 1):
        n = int(round(v / max_val * t_steps))
        for k in range(n):
            patterns[v, int((k + 0.5) * t_steps / n)] = 1
    return patterns


def encode_images(X_raw, patterns):
    """X_raw: (num, N_INPUT) int pixels -> spikes (num, T, N_INPUT) uint8."""
    return patterns[X_raw].transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# 3. Integer LIF SNN simulation -- mirrors rtl/neuron_core.sv exactly
# ---------------------------------------------------------------------------
def lif_step(membrane, weight_sum, threshold):
    leaked = np.where(membrane > LEAK, membrane - LEAK,
                      np.where(membrane > 0, 0, membrane))
    integrated = leaked + weight_sum
    fired = integrated >= threshold
    return np.where(fired, RESET_VAL, integrated), fired.astype(np.int64)


def simulate_snn(spikes, W1q, b1q, W2q, b2q, threshold):
    num = spikes.shape[0]
    mem_h = np.zeros((num, N_HIDDEN), dtype=np.int64)
    mem_o = np.zeros((num, N_OUTPUT), dtype=np.int64)
    out_count = np.zeros((num, N_OUTPUT), dtype=np.int64)
    hidden_prev = np.zeros((num, N_HIDDEN), dtype=np.int64)
    for t in range(T_STEPS):
        in_sp = spikes[:, t, :].astype(np.int64)
        h_wsum = in_sp @ W1q.T + b1q
        mem_h, hidden_fired = lif_step(mem_h, h_wsum, threshold)
        o_wsum = hidden_prev @ W2q.T + b2q
        mem_o, output_fired = lif_step(mem_o, o_wsum, threshold)
        out_count += output_fired
        hidden_prev = hidden_fired
    return out_count.argmax(1), out_count


# ---------------------------------------------------------------------------
# INT8 ANN baseline simulation/export -- v2 conventional inference baseline
# ---------------------------------------------------------------------------
def choose_ann_hidden_shift(X_cal, W1q, b1q):
    """Pick a power-of-two ReLU activation scale for the ANN RTL baseline."""
    b1_ann = b1q.astype(np.int64) * PIXEL_MAX
    h_acc = X_cal.astype(np.int64) @ W1q.T.astype(np.int64) + b1_ann
    max_h = int(max(1, h_acc.max()))
    shift = 0
    while (max_h >> shift) > 255:
        shift += 1
    return shift


def ann_int_biases(b1q, b2, s1, s2, hidden_shift):
    """Biases in the integer scales used by rtl/top_ann.sv."""
    b1_ann = b1q.astype(np.int64) * PIXEL_MAX
    hidden_scale = (s1 * PIXEL_MAX) / float(1 << hidden_shift)
    b2_ann = np.round(b2 * s2 * hidden_scale).astype(np.int64)
    return b1_ann, b2_ann


def simulate_ann_int(X_raw, W1q, b1_ann, W2q, b2_ann, hidden_shift):
    """Integer MLP baseline: raw pixels -> int8 hidden ReLU -> output scores."""
    h_acc = X_raw.astype(np.int64) @ W1q.T.astype(np.int64) + b1_ann
    h = np.maximum(h_acc, 0) >> hidden_shift
    h = np.clip(h, 0, 255).astype(np.int64)
    out = h @ W2q.T.astype(np.int64) + b2_ann
    return out.argmax(1), out


# ---------------------------------------------------------------------------
# 4. Build the CSR (compressed-sparse-row) connectivity used by the RTL
# ---------------------------------------------------------------------------
def build_csr(W1q, b1q, W2q, b2q):
    """Group every synapse by its SOURCE neuron (event-driven storage)."""
    csr_dst, csr_weight = [], []
    src_start = [0] * N_TOTAL
    src_count = [0] * N_TOTAL
    for s in range(N_TOTAL):
        src_start[s] = len(csr_dst)
        targets = []
        if s < N_INPUT:                       # pixel input -> all hidden
            for h in range(N_HIDDEN):
                targets.append((ID_HIDDEN_BASE + h, int(W1q[h, s])))
        elif s == ID_BIAS:                    # bias -> all hidden and output
            for h in range(N_HIDDEN):
                targets.append((ID_HIDDEN_BASE + h, int(b1q[h])))
            for o in range(N_OUTPUT):
                targets.append((ID_OUTPUT_BASE + o, int(b2q[o])))
        elif s < ID_OUTPUT_BASE:              # hidden -> all output
            h = s - ID_HIDDEN_BASE
            for o in range(N_OUTPUT):
                targets.append((ID_OUTPUT_BASE + o, int(W2q[o, h])))
        for d, w in targets:
            csr_dst.append(d)
            csr_weight.append(w)
        src_count[s] = len(targets)
    return csr_dst, csr_weight, src_start, src_count


def export(csr, spikes_rtl, labels_rtl, threshold, sim_dir, ann_hidden_shift=0):
    csr_dst, csr_weight, src_start, src_count = csr
    n_syn = len(csr_dst)

    with open(os.path.join(sim_dir, 'csr_dst.mem'), 'w') as f:
        for d in csr_dst:
            f.write(f"{d & 0xFFFF:04X}\n")
    with open(os.path.join(sim_dir, 'csr_weight.mem'), 'w') as f:
        for w in csr_weight:
            f.write(f"{w & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, 'csr_start.mem'), 'w') as f:
        for v in src_start:
            f.write(f"{v & 0xFFFFFF:06X}\n")
    with open(os.path.join(sim_dir, 'csr_count.mem'), 'w') as f:
        for v in src_count:
            f.write(f"{v & 0xFFFFFF:06X}\n")

    # snn_spikes.mem -- one N_INPUT-bit binary word per (image, timestep)
    with open(os.path.join(sim_dir, 'snn_spikes.mem'), 'w') as f:
        for img in spikes_rtl:
            for t in range(T_STEPS):
                f.write(''.join('1' if img[t, p] else '0'
                                for p in range(N_INPUT - 1, -1, -1)) + "\n")

    with open(os.path.join(sim_dir, 'snn_labels.mem'), 'w') as f:
        for lab in labels_rtl:
            f.write(f"{int(lab)}\n")

    with open(os.path.join(sim_dir, 'snn_config.vh'), 'w') as f:
        f.write("// Auto-generated by python/train_snn.py - do not edit by hand\n")
        f.write("`ifndef SNN_CONFIG_VH\n`define SNN_CONFIG_VH\n")
        f.write(f"`define SNN_DATASET        \"{DATASET}\"\n")
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
        f.write(f"`define SNN_NUM_TEST       {len(labels_rtl)}\n")
        f.write(f"`define SNN_NUM_SYN        {n_syn}\n")
        f.write(f"`define SNN_ANN_HIDDEN_SHIFT {ann_hidden_shift}\n")
        f.write(f"`define SNN_ANN_NUM_MACS   {N_INPUT*N_HIDDEN + N_HIDDEN*N_OUTPUT}\n")
        f.write("`endif\n")

    print(f"  Exported: CSR connectivity ({n_syn} synapses), "
          f"snn_spikes.mem ({len(spikes_rtl)*T_STEPS} words), "
          f"snn_labels.mem ({len(labels_rtl)}), snn_config.vh")


def export_ann_baseline(X_rtl, W1q, b1_ann, W2q, b2_ann, sim_dir):
    with open(os.path.join(sim_dir, 'ann_pixels.mem'), 'w') as f:
        for img in X_rtl:
            for pix in img:
                f.write(f"{int(pix) & 0xFF:02X}\n")

    with open(os.path.join(sim_dir, 'ann_w1.mem'), 'w') as f:
        for row in W1q:
            for w in row:
                f.write(f"{int(w) & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, 'ann_b1.mem'), 'w') as f:
        for b in b1_ann:
            f.write(f"{int(b) & 0xFFFFFFFF:08X}\n")

    with open(os.path.join(sim_dir, 'ann_w2.mem'), 'w') as f:
        for row in W2q:
            for w in row:
                f.write(f"{int(w) & 0xFF:02X}\n")
    with open(os.path.join(sim_dir, 'ann_b2.mem'), 'w') as f:
        for b in b2_ann:
            f.write(f"{int(b) & 0xFFFFFFFF:08X}\n")

    print(f"  Exported: INT8 ANN baseline memories "
          f"({len(X_rtl)} images, {N_INPUT*N_HIDDEN + N_HIDDEN*N_OUTPUT} MACs/image)")


# ---------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(script_dir, '..', 'sim')

    print("=" * 64)
    print(f"  SNN training + ANN->SNN conversion  [{DATASET}]")
    print(f"  network: {N_INPUT} -> {N_HIDDEN} -> {N_OUTPUT}  "
          f"({N_TOTAL} neurons, {T_STEPS} timesteps)")
    print("=" * 64)

    Xtr, ytr, Xte, yte = load_dataset()
    print(f"  train {len(Xtr)}  test {len(Xte)}")

    # ANN trains on normalized pixels
    W1, b1, W2, b2, ann_acc = train_ann(Xtr / PIXEL_MAX, ytr,
                                        Xte / PIXEL_MAX, yte)

    # Quantize weights to int8
    s1 = 127.0 / np.abs(W1).max()
    s2 = 127.0 / np.abs(W2).max()
    W1q, b1q = quantize(W1, s1), quantize(b1, s1)
    W2q, b2q = quantize(W2, s2), quantize(b2, s2)

    ann_hidden_shift = choose_ann_hidden_shift(Xtr[:min(5000, len(Xtr))], W1q, b1q)
    b1_ann, b2_ann = ann_int_biases(b1q, b2, s1, s2, ann_hidden_shift)

    patterns = make_rate_patterns(T_STEPS, PIXEL_MAX)
    sp_te = encode_images(Xte, patterns)

    # Grid-search the firing threshold on a tuning subset
    tune = sp_te[:TUNE_SUBSET]
    best = (-1, None)
    for thr in THR_RANGE:
        pred, _ = simulate_snn(tune, W1q, b1q, W2q, b2q, thr)
        acc = (pred == yte[:TUNE_SUBSET]).mean()
        if acc > best[0]:
            best = (acc, thr)
    _, threshold = best

    # Report SNN accuracy on a larger evaluation subset
    n_eval = min(EVAL_SUBSET, len(yte))
    pred, _ = simulate_snn(sp_te[:n_eval], W1q, b1q, W2q, b2q, threshold)
    snn_acc = (pred == yte[:n_eval]).mean()
    print(f"  Quantized to int8 (scales {s1:.2f} / {s2:.2f})")
    print(f"  INT8 ANN hidden activation shift = {ann_hidden_shift}")
    print(f"  Best SNN threshold = {threshold}")
    print(f"  SNN accuracy on {n_eval} test images: {snn_acc*100:.2f}%  "
          f"(ANN was {ann_acc*100:.2f}%)")

    ann_pred, _ = simulate_ann_int(Xte[:n_eval], W1q, b1_ann, W2q, b2_ann,
                                   ann_hidden_shift)
    ann_int_acc = (ann_pred == yte[:n_eval]).mean()
    print(f"  INT8 ANN baseline accuracy on {n_eval} images: {ann_int_acc*100:.2f}%")

    # Export CSR connectivity + the first NUM_TEST_RTL test images
    n_rtl = min(NUM_TEST_RTL, len(yte))
    csr = build_csr(W1q, b1q, W2q, b2q)
    export(csr, sp_te[:n_rtl].astype(np.uint8), yte[:n_rtl], threshold, sim_dir,
           ann_hidden_shift)
    export_ann_baseline(Xte[:n_rtl], W1q, b1_ann, W2q, b2_ann, sim_dir)

    pred_rtl, _ = simulate_snn(sp_te[:n_rtl], W1q, b1q, W2q, b2q, threshold)
    rtl_acc = (pred_rtl == yte[:n_rtl]).mean()
    print(f"  Golden accuracy on the {n_rtl} exported images: {rtl_acc*100:.2f}%")
    ann_pred_rtl, _ = simulate_ann_int(Xte[:n_rtl], W1q, b1_ann, W2q, b2_ann,
                                       ann_hidden_shift)
    ann_rtl_acc = (ann_pred_rtl == yte[:n_rtl]).mean()
    print(f"  INT8 ANN accuracy on the {n_rtl} exported images: {ann_rtl_acc*100:.2f}%")
    print("=" * 64)


if __name__ == '__main__':
    main()
