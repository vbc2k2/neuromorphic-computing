# Benchmark Tools

Use `tools/bench.py` as the main entry point for exports, Verilator runs, and
summary tables.

On HPRC:

```bash
source tools/env_hprc.sh
python3 tools/bench.py doctor
```

List benchmarks:

```bash
python3 tools/bench.py list
```

Run a complete benchmark:

```bash
python3 tools/bench.py all sparse
python3 tools/bench.py all nmnist
```

Run step by step:

```bash
python3 tools/bench.py export nmnist
python3 tools/bench.py run nmnist --design event --last 49
python3 tools/bench.py run nmnist --design ann --last 49
python3 tools/bench.py summarize nmnist
```

Pass exporter tuning parameters:

```bash
python3 tools/bench.py export nmnist --set NMNIST_EPOCHS=25 --set NMNIST_TOPK_W1=96
python3 tools/bench.py export shd --set SHD_N_HIDDEN=128 --set SHD_EPOCHS=50
```

Current benchmark targets:

- `mnist`: frame MNIST rate-coded SNN sanity check.
- `sparse`: controlled sparse-event microbenchmark.
- `shd`: public spike-audio dataset; hard for the current count model.
- `nmnist`: public event-camera digit dataset.
