# Server Run — MNIST SNN Benchmark (Xcelium + SLURM)

The MNIST network (784→256→10, ~203k synapses) is far too heavy for the local
xsim flow. This directory holds the scripts to compile and run it on a compute
cluster with **Cadence Xcelium**, with the test set split across cores.

## Steps

1. **Generate the MNIST inputs** (on any machine with Python):
   ```bash
   # python/train_snn.py has DATASET = 'mnist'
   python python/train_snn.py
   ```
   This writes into `sim/`: `csr_*.mem`, `snn_spikes.mem`, `snn_labels.mem`,
   `snn_config.vh`. Copy the whole repo to the cluster.

2. **Submit the array job** (6 slices of the test set, both designs per slice):
   ```bash
   sbatch server/sbatch_mnist.sh
   ```
   Adjust the `#SBATCH` directives and the `module load` line for your cluster.

3. **Merge the per-slice results** once all array tasks finish:
   ```bash
   python python/merge_results.py
   ```
   Produces `results/mnist_comparison.png` and `results/mnist_comparison.csv`.

## V2: add the INT8 ANN baseline

After regenerating exports with `python/train_snn.py`, run the three-way v2
comparison:

```bash
sbatch server/sbatch_mnist_v2.sh
python3 python/merge_results_v2.py
```

This runs event SNN, dense SNN, and the conventional INT8 ANN baseline for each
slice.

## Running a single slice by hand

```bash
bash server/xcelium_run.sh event 0 49 _0     # event design, images 0..49
bash server/xcelium_run.sh dense 0 49 _0     # dense baseline, images 0..49
```

`xcelium_run.sh` runs each design in its own `sim/run_<design><tag>/` directory
so concurrent slices do not collide. Output CSVs are copied back to `sim/`.

## Notes

- The testbenches accept `+first=N +last=M +tag=S` plusargs — that is how a
  slice picks its image range and names its output files.
- `SNN_NUM_TEST` in `sim/snn_config.vh` is how many test images were exported;
  keep `NUM_TEST` in `sbatch_mnist.sh` consistent with it.
- The event-driven design is much faster than the dense baseline; if you are
  core-limited, give the dense slices more wall-time.
