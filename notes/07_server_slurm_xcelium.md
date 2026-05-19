# Chapter 7: Server, SLURM, And Xcelium

## 7.1 Why The Server Is Needed

The MNIST network is much larger than the small digits network:

```text
784 -> 256 -> 10
```

It has about 203k synapses. The dense baseline scans all of them every timestep.

For 300 exported images and 50 timesteps per image, that is a lot of simulation
work. The server flow uses Xcelium and SLURM to run it on compute nodes.

## 7.2 Login Node vs Compute Node

Many university clusters have:

- Login nodes for editing, compiling small things, and submitting jobs.
- Compute nodes for heavy workloads.

Long simulations should run through SLURM, not directly on the login node.

## 7.3 What SLURM Does

SLURM is a job scheduler.

You submit a job:

```bash
sbatch server/sbatch_mnist.sh
```

SLURM puts it in a queue, assigns compute resources, and writes logs.

## 7.4 Array Jobs

An array job runs the same script multiple times with different array IDs.

This project uses:

```bash
#SBATCH --array=0-5
```

That creates six tasks:

```text
0, 1, 2, 3, 4, 5
```

Each task handles one slice of images.

## 7.5 Image Slicing

For 300 images and 6 slices:

```text
STRIDE = 50
```

So:

```text
task 0 -> images 0..49
task 1 -> images 50..99
task 2 -> images 100..149
task 3 -> images 150..199
task 4 -> images 200..249
task 5 -> images 250..299
```

Each task runs both designs:

```bash
bash server/xcelium_run.sh event FIRST LAST TAG
bash server/xcelium_run.sh dense FIRST LAST TAG
```

## 7.6 Why `SLURM_SUBMIT_DIR` Matters

SLURM may execute a copied version of your batch script from a spool directory.

If the script computes paths from its own location, it may accidentally look
inside:

```text
/var/spool/...
```

instead of your repository.

The script uses `SLURM_SUBMIT_DIR` so paths resolve from the directory where
you submitted the job.

Recommended submit location:

```bash
cd ~/neuromorphic_computing
sbatch server/sbatch_mnist.sh
```

## 7.7 Xcelium

Xcelium is Cadence's simulator. The command used here is:

```bash
xrun
```

`server/xcelium_run.sh` compiles and runs one design for one slice.

It chooses sources based on design:

```text
event -> top.sv + tb_classify.sv
dense -> top_dense.sv + tb_classify_dense.sv
```

## 7.8 Run Directories

Each invocation runs in a separate directory:

```text
sim/run_event_0/
sim/run_dense_0/
sim/run_event_1/
sim/run_dense_1/
...
```

This prevents parallel SLURM tasks from overwriting each other's simulator
work files.

## 7.9 Symbolic Links

The run script links generated memory files into each run directory:

```bash
ln -sf "$SIM/$f" .
```

The testbenches read simple relative names like:

```text
snn_spikes.mem
csr_dst.mem
```

The symlinks make those names available inside the run directory.

## 7.10 Logs

SLURM writes logs like:

```text
snn_mnist_JOBID_ARRAYID.log
```

Xcelium also writes logs inside each run directory:

```text
sim/run_event_0/xrun_event_0.log
sim/run_dense_0/xrun_dense_0.log
```

Check both when debugging.

## 7.11 Queue Status

Use:

```bash
squeue -u $USER
```

Common states:

```text
R  -> running
PD -> pending
CG -> completing
```

If a task is pending with `QOSMaxCpuPerUserLimit`, the cluster is limiting how
many CPUs you can use. It usually starts after another job finishes.

## 7.12 After The Jobs Finish

Check for output files:

```bash
ls sim/classify_*_*.csv
ls sim/metrics_classify_*_*.csv
```

For six slices and two designs, expect:

```text
12 classify CSVs
12 metrics CSVs
```

Then merge:

```bash
python3 python/merge_results.py
```

## 7.13 Useful Commands

Watch one log:

```bash
tail -f snn_mnist_436463_0.log
```

Search all logs for important lines:

```bash
grep -E "Classification accuracy|Synapse deliveries|Synapse ops|done|ERROR|readmem" snn_mnist_*.log
```

Cancel one job:

```bash
scancel JOBID
```

Cancel an array:

```bash
scancel 436463
```

## 7.14 Exercise

1. Why does the project use a SLURM array?
2. What does each array task run?
3. Why are separate run directories useful?
4. What does `QOSMaxCpuPerUserLimit` mean?
5. When should you run `merge_results.py`?

