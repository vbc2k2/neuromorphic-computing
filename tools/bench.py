#!/usr/bin/env python3
"""
bench.py - Small benchmark runner for the SNN/ANN RTL flows.

Examples:
    python3 tools/bench.py list
    python3 tools/bench.py doctor

    python3 tools/bench.py export sparse
    python3 tools/bench.py run sparse --design all
    python3 tools/bench.py summarize sparse

    python3 tools/bench.py all nmnist
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"


@dataclass(frozen=True)
class Bench:
    name: str
    dataset_id: str
    export_script: str
    default_tag: str
    description: str
    deps: tuple[str, ...] = ()


BENCHES: Dict[str, Bench] = {
    "mnist": Bench(
        name="mnist",
        dataset_id="mnist",
        export_script="python/train_snn.py",
        default_tag="_mnist",
        description="Frame MNIST rate-coded SNN sanity benchmark",
        deps=("torch", "numpy"),
    ),
    "sparse": Bench(
        name="sparse",
        dataset_id="sparse_event",
        export_script="python/export_sparse_event.py",
        default_tag="_sparse",
        description="Controlled sparse-event microbenchmark",
        deps=("numpy",),
    ),
    "shd": Bench(
        name="shd",
        dataset_id="shd",
        export_script="python/export_shd.py",
        default_tag="_shd",
        description="Public SHD spike-audio benchmark export",
        deps=("tonic", "torch", "numpy"),
    ),
    "nmnist": Bench(
        name="nmnist",
        dataset_id="nmnist",
        export_script="python/export_nmnist.py",
        default_tag="_nmnist",
        description="Public N-MNIST event-vision benchmark export",
        deps=("tonic", "torch", "numpy"),
    ),
}


def run(cmd: List[str], env: Optional[dict] = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def parse_key_values(values: Iterable[str]) -> dict:
    env = os.environ.copy()
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def parse_config(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or (SIM / "snn_config.vh")
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text().splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 3 and parts[0] == "`define":
            value = parts[2].strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            cfg[parts[1]] = value
    return cfg


def cfg_int(cfg: Dict[str, str], key: str, fallback: int = 0) -> int:
    try:
        return int(cfg.get(key, fallback))
    except ValueError:
        return fallback


def dense_ann_weight_count(cfg: Dict[str, str]) -> int:
    direct = cfg_int(cfg, "SNN_ANN_NUM_MACS")
    if direct > 0:
        return direct
    n_input = cfg_int(cfg, "SNN_N_INPUT")
    n_hidden = cfg_int(cfg, "SNN_N_HIDDEN")
    n_output = cfg_int(cfg, "SNN_N_OUTPUT")
    if n_input > 0 and n_hidden > 0 and n_output > 0:
        return n_input * n_hidden + n_hidden * n_output
    return 0


def require_bench(name: str) -> Bench:
    try:
        return BENCHES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BENCHES))
        raise SystemExit(f"Unknown benchmark '{name}'. Choices: {choices}") from exc


def check_import(module: str) -> bool:
    cmd = [sys.executable, "-c", f"import {module}"]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def read_metrics(path: Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row["value"]
            try:
                value_obj: object = int(value)
            except ValueError:
                try:
                    value_obj = float(value)
                except ValueError:
                    value_obj = value
            metrics[row["metric"]] = value_obj
    return metrics


def metric_path(design: str, tag: str) -> Path:
    return SIM / f"metrics_classify_{design}{tag}.csv"


def config_snapshot_path(tag: str) -> Path:
    return SIM / f"snn_config{tag or '_default'}.vh"


def config_for_tag(tag: str) -> tuple[Dict[str, str], str]:
    snapshot = config_snapshot_path(tag)
    if snapshot.exists():
        return parse_config(snapshot), snapshot.name
    return parse_config(), "snn_config.vh"


def current_dataset() -> str:
    return parse_config().get("SNN_DATASET", "")


def check_dataset_matches(bench: Bench, allow_mismatch: bool = False) -> None:
    dataset = current_dataset()
    if not dataset:
        raise SystemExit("sim/snn_config.vh is missing. Run export first.")
    if dataset != bench.dataset_id and not allow_mismatch:
        raise SystemExit(
            f"sim/ currently contains dataset '{dataset}', not '{bench.dataset_id}'. "
            f"Run: python3 tools/bench.py export {bench.name}"
        )


def command_list(_args: argparse.Namespace) -> None:
    print("Available benchmarks:")
    for bench in BENCHES.values():
        print(f"  {bench.name:<8} {bench.dataset_id:<13} {bench.description}")


def command_doctor(args: argparse.Namespace) -> None:
    print(f"repo: {ROOT}")
    print(f"python: {sys.executable}")
    print(f"bash: {shutil.which('bash') or 'MISSING'}")
    print(f"verilator: {shutil.which('verilator') or 'MISSING'}")
    if shutil.which("verilator"):
        subprocess.run(["verilator", "--version"], cwd=ROOT, check=False)

    cfg = parse_config()
    if cfg:
        print("sim/snn_config.vh:")
        for key in [
            "SNN_DATASET",
            "SNN_SELECTION",
            "SNN_LABEL_HIST",
            "SNN_N_INPUT",
            "SNN_N_HIDDEN",
            "SNN_N_OUTPUT",
            "SNN_T_STEPS",
            "SNN_NUM_TEST",
            "SNN_NUM_SYN",
            "SNN_TOPK_W1",
            "SNN_FINETUNE_EPOCHS",
        ]:
            if key in cfg:
                print(f"  {key}={cfg[key]}")
    else:
        print("sim/snn_config.vh: missing")

    benches = [require_bench(args.benchmark)] if args.benchmark else BENCHES.values()
    for bench in benches:
        missing = [dep for dep in bench.deps if not check_import(dep)]
        if missing:
            print(f"{bench.name}: missing Python deps: {', '.join(missing)}")
        else:
            print(f"{bench.name}: Python deps OK")


def command_export(args: argparse.Namespace) -> None:
    bench = require_bench(args.benchmark)
    script = ROOT / bench.export_script
    if not script.exists():
        raise SystemExit(f"Missing exporter: {script}")
    env = parse_key_values(args.set)
    run([sys.executable, bench.export_script], env=env)
    cfg = parse_config()
    print("\nExported config:")
    print(f"  dataset={cfg.get('SNN_DATASET', '?')}")
    print(f"  test={cfg.get('SNN_NUM_TEST', '?')}")
    print(f"  synapses={cfg.get('SNN_NUM_SYN', '?')}")


def command_run(args: argparse.Namespace) -> None:
    bench = require_bench(args.benchmark)
    check_dataset_matches(bench, args.allow_mismatch)
    if not shutil.which("bash"):
        raise SystemExit("bash is not on PATH")
    if not shutil.which("verilator"):
        raise SystemExit(
            "verilator is not on PATH. On HPRC run: source tools/env_hprc.sh"
        )

    cfg = parse_config()
    first = args.first
    if args.last is None:
        last = int(cfg.get("SNN_NUM_TEST", "300")) - 1
    else:
        last = args.last
    tag = args.tag or bench.default_tag
    config_snapshot_path(tag).write_text((SIM / "snn_config.vh").read_text())

    designs = ["event", "ann"] if args.design == "all" else [args.design]
    for design in designs:
        if design == "event":
            run([
                "bash",
                "tools/run_verilator_event.sh",
                str(first),
                str(last),
                tag,
                str(args.max_cycles),
            ])
        elif design == "event_ram":
            run([
                "bash",
                "tools/run_verilator_event_ram.sh",
                str(first),
                str(last),
                tag,
                str(args.max_cycles),
            ])
        elif design == "event_ram2":
            run([
                "bash",
                "tools/run_verilator_event_ram2.sh",
                str(first),
                str(last),
                tag,
                str(args.max_cycles),
            ])
        elif design == "ann":
            run(["bash", "tools/run_verilator_ann.sh", str(first), str(last), tag])
        else:
            raise SystemExit(f"Unsupported design: {design}")


def summarize_rows(tag: str) -> List[tuple[str, Dict[str, object]]]:
    rows: List[tuple[str, Dict[str, object]]] = []
    for design in ["event", "ann"]:
        path = metric_path(design, tag)
        if path.exists():
            rows.append((design, read_metrics(path)))
    return rows


def command_summarize(args: argparse.Namespace) -> None:
    bench = require_bench(args.benchmark) if args.benchmark else None
    tag = args.tag or (bench.default_tag if bench else "")
    rows = summarize_rows(tag)
    if not rows:
        raise SystemExit(f"No metrics found for tag '{tag}' in sim/")
    cfg, cfg_source = config_for_tag(tag)
    if bench and cfg.get("SNN_DATASET") and cfg.get("SNN_DATASET") != bench.dataset_id:
        raise SystemExit(
            f"tag '{tag}' uses dataset '{cfg.get('SNN_DATASET')}', not "
            f"'{bench.dataset_id}'. Use the matching benchmark/tag."
        )

    print("=" * 78)
    print(f"Benchmark summary tag={tag}")
    print(f"Config source: {cfg_source}")
    if cfg.get("SNN_DATASET"):
        print(f"Dataset: {cfg.get('SNN_DATASET')}")
    if cfg.get("SNN_SELECTION"):
        print(f"Selection: {cfg.get('SNN_SELECTION')}")
    if cfg.get("SNN_LABEL_HIST"):
        print(f"Label histogram: {cfg.get('SNN_LABEL_HIST')}")
    print("=" * 78)
    for design, metrics in rows:
        ops = metrics.get("mac_ops", metrics.get("synapse_ops", 0))
        print(
            f"{design:<8} images={int(metrics.get('num_images', 0)):>5} "
            f"acc={float(metrics.get('accuracy_pct', 0.0)):>7.2f}% "
            f"active={int(metrics.get('active_cycles', 0)):>14,} "
            f"ops={int(ops):>14,}"
        )

    lookup = dict(rows)
    if "event" in lookup and "ann" in lookup:
        event = lookup["event"]
        ann = lookup["ann"]
        if int(event.get("num_images", 0)) != int(ann.get("num_images", 0)):
            raise SystemExit(
                "Refusing ratio: event and ANN runs used different image counts "
                f"({event.get('num_images')} vs {ann.get('num_images')})."
            )
        if (
            "first_image" in event
            and "first_image" in ann
            and (
                int(event.get("first_image", -1)) != int(ann.get("first_image", -2))
                or int(event.get("last_image", -1)) != int(ann.get("last_image", -2))
            )
        ):
            raise SystemExit(
                "Refusing ratio: event and ANN runs used different image ranges "
                f"({event.get('first_image')}..{event.get('last_image')} vs "
                f"{ann.get('first_image')}..{ann.get('last_image')})."
            )
        ev_active = max(int(event.get("active_cycles", 0)), 1)
        ev_ops = max(int(event.get("synapse_ops", 0)), 1)
        ann_active = int(ann.get("active_cycles", 0))
        ann_ops = int(ann.get("mac_ops", ann.get("synapse_ops", 0)))
        print("-" * 78)
        print(f"ANN/Event active-cycle ratio: {ann_active / ev_active:.2f}x")
        print(f"ANN/Event op ratio:           {ann_ops / ev_ops:.2f}x")
        sparse_weights = cfg_int(cfg, "SNN_NUM_SYN")
        dense_weights = dense_ann_weight_count(cfg)
        if sparse_weights > 0 and dense_weights > 0:
            print(
                f"Dense/SNN weight-storage ratio: {dense_weights / sparse_weights:.2f}x "
                f"({dense_weights:,} dense weights vs {sparse_weights:,} CSR synapses)"
            )

        # First-order proxy only: a real claim still needs synthesis plus activity.
        # It is useful because edge energy is usually dominated by memory traffic
        # and MAC count, not by Verilator wall-clock runtime.
        equal_cost_ratio = ann_ops / ev_ops
        mac_weighted_ratio = (ann_ops * 3.0) / ev_ops
        print(f"Equal-cost work proxy ratio:   {equal_cost_ratio:.2f}x")
        print(f"3x-MAC work proxy ratio:       {mac_weighted_ratio:.2f}x")
    print("=" * 78)


def command_all(args: argparse.Namespace) -> None:
    command_export(args)
    run_args = argparse.Namespace(
        benchmark=args.benchmark,
        first=args.first,
        last=args.last,
        tag=args.tag,
        design=args.design,
        max_cycles=args.max_cycles,
        allow_mismatch=False,
    )
    command_run(run_args)
    summary_args = argparse.Namespace(benchmark=args.benchmark, tag=args.tag)
    command_summarize(summary_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plug-and-play benchmark runner for the SNN/ANN RTL flows"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available benchmarks").set_defaults(func=command_list)

    doctor = sub.add_parser("doctor", help="check tools, Python deps, and sim config")
    doctor.add_argument("benchmark", nargs="?")
    doctor.set_defaults(func=command_doctor)

    export = sub.add_parser("export", help="export sim/*.mem for a benchmark")
    export.add_argument("benchmark")
    export.add_argument("--set", action="append", default=[],
                        help="set environment variable for exporter, e.g. --set NMNIST_EPOCHS=25")
    export.set_defaults(func=command_export)

    run_p = sub.add_parser("run", help="run Verilator RTL for current exported benchmark")
    run_p.add_argument("benchmark")
    run_p.add_argument("--design", choices=["event", "event_ram", "event_ram2", "ann", "all"], default="all")
    run_p.add_argument("--first", "--start", dest="first", type=int, default=0)
    run_p.add_argument("--last", "--end", dest="last", type=int)
    run_p.add_argument("--tag")
    run_p.add_argument("--max-cycles", type=int, default=50000000)
    run_p.add_argument("--allow-mismatch", action="store_true")
    run_p.set_defaults(func=command_run)

    summary = sub.add_parser("summarize", help="summarize sim metrics CSVs")
    summary.add_argument("benchmark", nargs="?")
    summary.add_argument("--tag")
    summary.set_defaults(func=command_summarize)

    all_p = sub.add_parser("all", help="export, run event+ANN, and summarize")
    all_p.add_argument("benchmark")
    all_p.add_argument("--design", choices=["event", "event_ram", "event_ram2", "ann", "all"], default="all")
    all_p.add_argument("--first", "--start", dest="first", type=int, default=0)
    all_p.add_argument("--last", "--end", dest="last", type=int)
    all_p.add_argument("--tag")
    all_p.add_argument("--max-cycles", type=int, default=50000000)
    all_p.add_argument("--set", action="append", default=[])
    all_p.set_defaults(func=command_all)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
