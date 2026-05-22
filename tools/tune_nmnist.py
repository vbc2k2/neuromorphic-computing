#!/usr/bin/env python3
"""
tune_nmnist.py - Export-only N-MNIST algorithm sweep driver.

This script does not run Verilator or Yosys for every point. It searches the
cheap algorithm/export space first, records proxy metrics, and leaves RTL runs
for the Pareto candidates.

Examples:
    python3 tools/tune_nmnist.py --preset quick
    python3 tools/tune_nmnist.py --preset edge --max-configs 8
    python3 tools/tune_nmnist.py --grid NMNIST_TOPK_W1=96,128 --grid NMNIST_BIAS_MODE=none,hidden
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"
RESULTS = ROOT / "results"


PRESETS: dict[str, dict[str, list[str]]] = {
    "quick": {
        "NMNIST_T_STEPS": ["50"],
        "NMNIST_TOPK_W1": ["96", "128"],
        "NMNIST_FINETUNE_EPOCHS": ["12"],
        "NMNIST_BIAS_MODE": ["none", "hidden"],
        "NMNIST_READOUT": ["membrane"],
        "NMNIST_PRUNE_MODE": ["per_hidden"],
        "NMNIST_ACTIVITY_LAMBDA": ["0"],
    },
    "edge": {
        "NMNIST_T_STEPS": ["40", "50"],
        "NMNIST_TOPK_W1": ["64", "96", "128"],
        "NMNIST_FINETUNE_EPOCHS": ["12"],
        "NMNIST_BIAS_MODE": ["none", "hidden"],
        "NMNIST_READOUT": ["membrane"],
        "NMNIST_PRUNE_MODE": ["per_hidden", "saliency"],
        "NMNIST_ACTIVITY_LAMBDA": ["0", "0.0001"],
    },
    "accuracy": {
        "NMNIST_T_STEPS": ["50", "75"],
        "NMNIST_TOPK_W1": ["128", "160", "192"],
        "NMNIST_FINETUNE_EPOCHS": ["12", "20"],
        "NMNIST_BIAS_MODE": ["hidden", "all"],
        "NMNIST_READOUT": ["membrane"],
        "NMNIST_PRUNE_MODE": ["per_hidden", "saliency"],
        "NMNIST_ACTIVITY_LAMBDA": ["0"],
    },
}


SUMMARY_FIELDS = [
    "tag",
    "status",
    "t_steps",
    "topk_w1",
    "finetune_epochs",
    "bias_mode",
    "readout",
    "prune_mode",
    "activity_lambda",
    "ann_int_acc_pct",
    "snn_eval_acc_pct",
    "accuracy_gap_pct",
    "csr_synapses",
    "dense_to_sparse_storage_ratio",
    "snn_eval_deliveries_per_sample",
    "dense_to_snn_delivery_ratio",
    "accuracy_per_ksynapse",
    "accuracy_per_kdelivery",
    "threshold",
    "ann_hidden_shift",
    "log",
]


def parse_grid(items: list[str]) -> dict[str, list[str]]:
    grid: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Bad --grid item '{item}', expected KEY=A,B,C")
        key, raw_values = item.split("=", 1)
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not key or not values:
            raise SystemExit(f"Bad --grid item '{item}', expected KEY=A,B,C")
        grid[key.strip()] = values
    return grid


def config_name(cfg: dict[str, str]) -> str:
    t = cfg.get("NMNIST_T_STEPS", "?")
    k = cfg.get("NMNIST_TOPK_W1", "?")
    e = cfg.get("NMNIST_FINETUNE_EPOCHS", "?")
    bias = cfg.get("NMNIST_BIAS_MODE", "none")
    readout = cfg.get("NMNIST_READOUT", "membrane")
    prune = cfg.get("NMNIST_PRUNE_MODE", "per_hidden")
    lam = cfg.get("NMNIST_ACTIVITY_LAMBDA", "0").replace(".", "p")
    return f"t{t}_k{k}_e{e}_b{bias}_r{readout}_p{prune}_a{lam}"


def iter_configs(grid: dict[str, list[str]]):
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, values))


def read_metrics() -> dict:
    path = SIM / "export_nmnist_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} was not produced")
    with path.open() as f:
        return json.load(f)


def flatten_metrics(tag: str, status: str, metrics: dict, log_path: Path) -> dict:
    dense_macs = float(metrics.get("dense_ann_macs_per_sample", 0.0))
    deliveries = float(metrics.get("snn_eval_deliveries_per_sample", 0.0))
    ann_acc = float(metrics.get("ann_int_acc_pct", 0.0))
    snn_acc = float(metrics.get("snn_eval_acc_pct", 0.0))
    row = {
        "tag": tag,
        "status": status,
        "t_steps": metrics.get("t_steps", ""),
        "topk_w1": metrics.get("topk_w1", ""),
        "finetune_epochs": metrics.get("finetune_epochs", ""),
        "bias_mode": metrics.get("bias_mode", ""),
        "readout": metrics.get("readout", ""),
        "prune_mode": metrics.get("prune_mode", ""),
        "activity_lambda": metrics.get("activity_lambda", ""),
        "ann_int_acc_pct": ann_acc,
        "snn_eval_acc_pct": snn_acc,
        "accuracy_gap_pct": ann_acc - snn_acc,
        "csr_synapses": metrics.get("csr_synapses", ""),
        "dense_to_sparse_storage_ratio": metrics.get("dense_to_sparse_storage_ratio", ""),
        "snn_eval_deliveries_per_sample": deliveries,
        "dense_to_snn_delivery_ratio": dense_macs / deliveries if deliveries > 0 else "",
        "accuracy_per_ksynapse": metrics.get("accuracy_per_ksynapse", ""),
        "accuracy_per_kdelivery": metrics.get("accuracy_per_kdelivery", ""),
        "threshold": metrics.get("threshold", ""),
        "ann_hidden_shift": metrics.get("ann_hidden_shift", ""),
        "log": str(log_path.relative_to(ROOT)),
    }
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def snapshot(tag: str, out_dir: Path) -> None:
    dest = out_dir / tag
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["snn_config.vh", "export_nmnist_metrics.json"]:
        src = SIM / name
        if src.exists():
            shutil.copy2(src, dest / name)


def is_dominated(row: dict, other: dict) -> bool:
    return (
        float(other["snn_eval_acc_pct"]) >= float(row["snn_eval_acc_pct"])
        and float(other["csr_synapses"]) <= float(row["csr_synapses"])
        and float(other["snn_eval_deliveries_per_sample"]) <= float(row["snn_eval_deliveries_per_sample"])
        and (
            float(other["snn_eval_acc_pct"]) > float(row["snn_eval_acc_pct"])
            or float(other["csr_synapses"]) < float(row["csr_synapses"])
            or float(other["snn_eval_deliveries_per_sample"]) < float(row["snn_eval_deliveries_per_sample"])
        )
    )


def pareto(rows: list[dict]) -> list[dict]:
    good = [row for row in rows if row.get("status") == "ok"]
    return [row for row in good if not any(is_dominated(row, other) for other in good)]


def print_table(title: str, rows: list[dict], limit: int) -> None:
    print("\n" + title)
    print("-" * len(title))
    header = "tag                         snn%   ann%   gap   synapses  deliv/img  storeX  opX"
    print(header)
    for row in rows[:limit]:
        print(
            f"{row['tag']:<27} "
            f"{float(row['snn_eval_acc_pct']):5.2f} "
            f"{float(row['ann_int_acc_pct']):6.2f} "
            f"{float(row['accuracy_gap_pct']):5.2f} "
            f"{int(row['csr_synapses']):8d} "
            f"{float(row['snn_eval_deliveries_per_sample']):9.1f} "
            f"{float(row['dense_to_sparse_storage_ratio']):6.2f} "
            f"{float(row['dense_to_snn_delivery_ratio']):5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export-only N-MNIST algorithm tuner")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--grid", action="append", default=[],
                        help="override/add grid values, e.g. NMNIST_TOPK_W1=96,128")
    parser.add_argument("--out", default="results/nmnist_tune/summary.csv")
    parser.add_argument("--tag-prefix", default="tune")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--snapshot", action="store_true",
                        help="copy config and export JSON for each completed point")
    parser.add_argument("--dry-run", action="store_true",
                        help="print expanded configurations without running exports")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    grid = {key: list(values) for key, values in PRESETS[args.preset].items()}
    grid.update(parse_grid(args.grid))

    out_csv = ROOT / args.out
    out_dir = out_csv.parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if args.resume and out_csv.exists():
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
    seen = {row["tag"] for row in rows}

    configs = list(iter_configs(grid))
    if args.max_configs is not None:
        configs = configs[:args.max_configs]

    if args.dry_run:
        for idx, cfg in enumerate(configs, start=1):
            tag = f"{args.tag_prefix}_{config_name(cfg)}"
            print(f"[{idx}/{len(configs)}] {tag}")
            print("  " + " ".join(f"{key}={value}" for key, value in sorted(cfg.items())))
        return

    for idx, cfg in enumerate(configs, start=1):
        tag = f"{args.tag_prefix}_{config_name(cfg)}"
        if tag in seen:
            continue

        env = os.environ.copy()
        env.update(cfg)
        log_path = log_dir / f"{tag}.log"
        cmd = [sys.executable, "python/export_nmnist.py"]
        print(f"[{idx}/{len(configs)}] {tag}")
        print("  " + " ".join(f"{key}={value}" for key, value in sorted(cfg.items())))

        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        if proc.returncode != 0:
            row = {"tag": tag, "status": f"failed:{proc.returncode}", "log": str(log_path.relative_to(ROOT))}
            rows.append(row)
            write_csv(out_csv, rows)
            print(f"  failed, see {log_path.relative_to(ROOT)}")
            continue

        metrics = read_metrics()
        row = flatten_metrics(tag, "ok", metrics, log_path)
        rows.append(row)
        write_csv(out_csv, rows)
        if args.snapshot:
            snapshot(tag, out_dir)
        if not args.quiet:
            print(
                f"  snn={float(row['snn_eval_acc_pct']):.2f}% "
                f"ann={float(row['ann_int_acc_pct']):.2f}% "
                f"syn={int(row['csr_synapses'])} "
                f"deliveries/img={float(row['snn_eval_deliveries_per_sample']):.1f}"
            )

    good = [row for row in rows if row.get("status") == "ok"]
    if not good:
        print(f"No successful points. See {out_csv.relative_to(ROOT)}")
        return

    print(f"\nWrote {out_csv.relative_to(ROOT)}")
    print_table("Top Accuracy", sorted(good, key=lambda r: float(r["snn_eval_acc_pct"]), reverse=True), args.show)
    print_table(
        "Top Accuracy Per Synapse",
        sorted(good, key=lambda r: float(r["accuracy_per_ksynapse"]), reverse=True),
        args.show,
    )
    print_table(
        "Top Accuracy Per Delivery",
        sorted(good, key=lambda r: float(r["accuracy_per_kdelivery"]), reverse=True),
        args.show,
    )
    print_table(
        "Pareto Frontier",
        sorted(pareto(good), key=lambda r: (float(r["snn_eval_acc_pct"]), -float(r["csr_synapses"])), reverse=True),
        args.show,
    )


if __name__ == "__main__":
    main()
