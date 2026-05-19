# Chapter 8: Debugging And Validating Results

## 8.1 First Rule

Do not trust a simulation just because it printed `done`.

A simulation can finish while still producing invalid results. Always check
logs and metrics.

## 8.2 Good Signs

A healthy run should have:

- No `$readmem error`.
- No compile errors.
- No missing file errors.
- Nonzero event deliveries.
- Nonzero dense synapse operations.
- Reasonable classification accuracy.
- Event and dense predictions matching.

## 8.3 Bad Sign: `$readmem error`

Example:

```text
$readmem error: number too long in file "csr_dst.mem" at line 1
```

Meaning:

The memory file contains a word wider than the RTL memory declaration.

Consequence:

CSR may load incorrectly, often as zeros.

Symptoms:

- Very low accuracy.
- Predictions mostly `0`.
- `Synapse ops: 0` or `Synapse deliveries: 0`.

Fix:

Make the RTL memory load width match the exported file format.

## 8.4 Bad Sign: Missing `xcelium_run.sh`

Example:

```text
bash: /var/spool/.../xcelium_run.sh: No such file or directory
```

Meaning:

The batch script derived paths from SLURM's copied spool script instead of the
repository.

Fix:

Anchor paths at `SLURM_SUBMIT_DIR`.

## 8.5 Bad Sign: `xrun` Not Found

Example:

```text
ERROR: xrun is not on PATH
```

Meaning:

The Cadence/Xcelium environment is not loaded for the job.

Check interactively:

```bash
xrun -version
```

If it works interactively but not in the job, the batch environment differs
from your shell environment.

## 8.6 Bad Sign: `[TIMEOUT]`

Example:

```text
[TIMEOUT] Simulation exceeded time limit
cp: cannot stat 'metrics_classify_event_4.csv': No such file or directory
```

Meaning:

The testbench watchdog stopped the simulation before it reached the normal
result-writing code.

This can happen even when the first printed image predictions look correct.
Those early predictions only prove the simulation started correctly. They do not
prove the slice finished.

Symptoms:

- Only the first few image predictions appear.
- No final classification accuracy block.
- No metrics CSV.
- The run script fails when copying the missing metrics file.

Fix:

Increase the testbench watchdog or pass a larger `+timeout_ns` plusarg. The
server run script uses `SNN_TIMEOUT_NS` to set this value.

Example:

```bash
SNN_TIMEOUT_NS=120000000000 bash server/xcelium_run.sh event 200 249 _4
```

## 8.7 Noisy But Usually Harmless: `cds.lib`

Example:

```text
cds.lib Invalid environment variable
```

This is usually a Cadence library configuration warning from your home
directory.

It is not the main problem if the design still compiles and simulates.

Focus first on:

- Errors.
- `$readmem` failures.
- Missing outputs.
- Zero metrics.

## 8.8 Checking Logs

Useful command:

```bash
grep -E "ERROR|FATAL|readmem|Classification accuracy|Synapse deliveries|Synapse ops|done" snn_mnist_*.log
```

Look for:

```text
Classification accuracy: ...
Event-driven work:
Dense-baseline work:
done
```

Also check the Xcelium logs inside `sim/run_*`.

## 8.9 Checking Output Files

For six slices:

```bash
ls sim/classify_event_*.csv
ls sim/classify_dense_*.csv
ls sim/metrics_classify_event_*.csv
ls sim/metrics_classify_dense_*.csv
```

Expected:

```text
6 event classify files
6 dense classify files
6 event metrics files
6 dense metrics files
```

Total:

```text
12 classify files
12 metrics files
```

## 8.10 Stale Output Files

Old CSVs can trick you.

If a new run fails before writing outputs, old CSVs might still remain.

Before rerunning:

```bash
rm -f snn_mnist_*.log sim/classify_*_*.csv sim/metrics_classify_*_*.csv
```

The run script now also removes stale slice outputs before each design run.

## 8.11 Interpreting Accuracy

There are several accuracy numbers:

1. ANN accuracy:
   - Floating-point PyTorch model.

2. Quantized SNN accuracy:
   - Python integer golden model.

3. RTL exported-image accuracy:
   - Hardware simulation on exported images.

The RTL accuracy should match the Python golden model for the same exported
images.

It does not need to match the floating-point ANN exactly.

## 8.12 Interpreting Work Metrics

Dense baseline:

```text
synapse_ops should be large
```

Event-driven:

```text
synapse_ops in metrics means router deliveries
```

If event-driven work is lower and predictions match, the accelerator is doing
what it is designed to do.

## 8.13 Event vs Dense Prediction Mismatch

If event and dense predictions differ:

1. Check that both loaded the same memory files.
2. Check `snn_config.vh`.
3. Check reset behavior.
4. Check output spike counting.
5. Check the timing of timestep tick and spike delivery.
6. Compare a small slice first.

Do not start performance analysis until correctness is fixed.

## 8.14 Minimal Debug Flow

When something goes wrong:

```text
1. Check whether Python export completed.
2. Check generated file timestamps.
3. Check `snn_config.vh`.
4. Run one slice manually.
5. Search log for errors.
6. Confirm CSVs were produced.
7. Compare event vs dense predictions.
8. Merge only after all slices are valid.
```

## 8.15 Useful One-Slice Run

Run only a small slice by hand:

```bash
bash server/xcelium_run.sh event 0 4 _debug
bash server/xcelium_run.sh dense 0 4 _debug
```

This is faster than launching the whole array when debugging.

## 8.16 Exercise

1. Why can a simulation print `done` but still be invalid?
2. What does a `$readmem` width error usually imply?
3. Why should stale CSVs be deleted before reruns?
4. Which accuracy should RTL match?
5. What must be true before comparing performance metrics?
