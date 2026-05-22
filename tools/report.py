#!/usr/bin/env python3
"""
report.py - Combine RTL metrics and Yosys stats for one benchmark tag.

Examples:
    python3 tools/report.py --tag _nmnist_t50_k128_bram --event-design event_ram
    python3 tools/report.py --tag _nmnist_t50_k128_bram --event-design event_ram --format md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"
RESULTS = ROOT / "results"


def parse_config(path: Path) -> Dict[str, str]:
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


def read_metrics(path: Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    if not path.exists():
        return metrics
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            value = row["value"]
            try:
                metrics[row["metric"]] = int(value)
            except ValueError:
                try:
                    metrics[row["metric"]] = float(value)
                except ValueError:
                    metrics[row["metric"]] = value
    return metrics


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
    return n_input * n_hidden + n_hidden * n_output


def metric_path(kind: str, tag: str) -> Path:
    return SIM / f"metrics_classify_{kind}{tag}.csv"


def config_path(tag: str) -> Path:
    tagged = SIM / f"snn_config{tag}.vh"
    if tagged.exists():
        return tagged
    return SIM / "snn_config.vh"


def yosys_path(kind: str, tag: str, flow: str) -> Path:
    return RESULTS / f"yosys_{kind}{tag}_{flow}.log"


def final_design_block(text: str) -> str:
    marker = "=== design hierarchy ==="
    if marker in text:
        return text.rsplit(marker, 1)[1]
    # Fall back to the last module block. This keeps partial logs useful.
    parts = text.split("===")
    return parts[-1] if parts else text


def parse_yosys(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}

    text = path.read_text(errors="replace")
    stats: Dict[str, int] = {"log_exists": 1}

    lc_matches = re.findall(r"Estimated number of LCs:\s*([0-9]+)", text)
    if lc_matches:
        stats["estimated_lcs"] = int(lc_matches[-1])

    block = final_design_block(text)
    in_cells = False
    for raw_line in block.splitlines():
        line = raw_line.strip()
        m_total = re.fullmatch(r"([0-9]+)\s+cells", line)
        if m_total:
            stats["cells"] = int(m_total.group(1))
            in_cells = True
            continue

        if not in_cells:
            continue
        if not line or line.startswith("Warnings:") or line.startswith("End of script"):
            break

        m = re.fullmatch(r"([0-9]+)\s+(\S+)", line)
        if not m:
            continue
        count = int(m.group(1))
        cell = m.group(2)
        if cell in {"submodules", "memories", "processes"}:
            continue
        if cell.startswith("$paramod") or cell in {"top", "top_ann", "top_event_ram"}:
            continue
        stats[cell] = stats.get(cell, 0) + count

    return stats


def get_any(stats: Dict[str, int], names: Iterable[str]) -> int:
    return sum(stats.get(name, 0) for name in names)


def ratio(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def fmt_num(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(fmt_num(value)))

    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(fmt_num(value).ljust(widths[i]) for i, value in enumerate(row)))


def print_markdown(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(fmt_num(value) for value in row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize accuracy, work, FPGA stats, and ASIC-style stats for a tag"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--event-design", choices=["event", "event_ram", "event_ram2"], default="event_ram")
    parser.add_argument("--format", choices=["text", "md", "csv"], default="text")
    args = parser.parse_args()

    tag = args.tag
    event_log_prefix = args.event_design
    event_display = args.event_design

    cfg = parse_config(config_path(tag))
    event_metrics = read_metrics(metric_path("event", tag))
    ann_metrics = read_metrics(metric_path("ann", tag))

    synth = {
        event_display: {
            "xilinx": parse_yosys(yosys_path(event_log_prefix, tag, "xilinx")),
            "asic": parse_yosys(yosys_path(event_log_prefix, tag, "asic")),
        },
        "ann": {
            "xilinx": parse_yosys(yosys_path("ann", tag, "xilinx")),
            "asic": parse_yosys(yosys_path("ann", tag, "asic")),
        },
    }

    if args.format == "text":
        print("=" * 86)
        print(f"Report tag={tag}")
        print(f"Config source: {config_path(tag).name}")
        for key in [
            "SNN_DATASET",
            "SNN_SELECTION",
            "SNN_LABEL_HIST",
            "SNN_T_STEPS",
            "SNN_TOPK_W1",
            "SNN_FINETUNE_EPOCHS",
            "SNN_BIAS_MODE",
            "SNN_READOUT",
            "SNN_ACTIVITY_LAMBDA",
            "SNN_NUM_SYN",
        ]:
            if key in cfg:
                print(f"{key}: {cfg[key]}")
        print("=" * 86)

    metric_rows: list[list[object]] = []
    for name, metrics in [(event_display, event_metrics), ("ann", ann_metrics)]:
        if not metrics:
            continue
        ops = metrics.get("mac_ops", metrics.get("synapse_ops", 0))
        metric_rows.append([
            name,
            metrics.get("num_images"),
            metrics.get("accuracy_pct"),
            metrics.get("active_cycles"),
            ops,
        ])

    synth_rows: list[list[object]] = []
    for name in [event_display, "ann"]:
        for flow in ["xilinx", "asic"]:
            stats = synth[name][flow]
            if not stats:
                continue
            if flow == "xilinx":
                synth_rows.append([
                    name,
                    flow,
                    stats.get("estimated_lcs"),
                    stats.get("cells"),
                    get_any(stats, ["DSP48E1"]),
                    get_any(stats, ["RAMB36E1"]),
                    get_any(stats, ["RAMB18E1"]),
                    get_any(stats, ["$mem_v2"]),
                ])
            else:
                synth_rows.append([
                    name,
                    flow,
                    stats.get("estimated_lcs"),
                    stats.get("cells"),
                    "-",
                    "-",
                    "-",
                    get_any(stats, ["$mem_v2"]),
                ])

    if args.format == "csv":
        writer = csv.writer(__import__("sys").stdout)
        writer.writerow(["section", "design", "field", "value"])
        for row in metric_rows:
            design, images, acc, active, ops = row
            writer.writerow(["metrics", design, "images", images])
            writer.writerow(["metrics", design, "accuracy_pct", acc])
            writer.writerow(["metrics", design, "active_cycles", active])
            writer.writerow(["metrics", design, "ops", ops])
        for row in synth_rows:
            design, flow, lcs, cells, dsp, ramb36, ramb18, memv2 = row
            for field, value in [
                ("estimated_lcs", lcs),
                ("cells", cells),
                ("dsp48e1", dsp),
                ("ramb36e1", ramb36),
                ("ramb18e1", ramb18),
                ("mem_v2", memv2),
            ]:
                writer.writerow([f"synth_{flow}", design, field, value])
        return

    emit_table = print_markdown if args.format == "md" else print_table

    if metric_rows:
        if args.format == "text":
            print("RTL Metrics")
        emit_table(["design", "images", "accuracy_pct", "active_cycles", "ops"], metric_rows)

    if synth_rows:
        if args.format == "text":
            print("\nSynthesis")
        emit_table(["design", "flow", "estimated_lcs", "cells", "dsp48", "ramb36", "ramb18", "mem_v2"], synth_rows)

    if event_metrics and ann_metrics and args.format == "text":
        ev_active = int(event_metrics.get("active_cycles", 0))
        ev_ops = int(event_metrics.get("synapse_ops", 0))
        ann_active = int(ann_metrics.get("active_cycles", 0))
        ann_ops = int(ann_metrics.get("mac_ops", ann_metrics.get("synapse_ops", 0)))
        sparse_weights = cfg_int(cfg, "SNN_NUM_SYN")
        dense_weights = dense_ann_weight_count(cfg)
        print("\nRatios")
        for label, value in [
            ("ANN/Event active-cycle ratio", ratio(ann_active, ev_active)),
            ("ANN/Event op ratio", ratio(ann_ops, ev_ops)),
            ("Dense/SNN weight-storage ratio", ratio(dense_weights, sparse_weights)),
        ]:
            if value is not None:
                print(f"{label}: {value:.2f}x")

        ev_x = synth[event_display]["xilinx"]
        ann_x = synth["ann"]["xilinx"]
        ev_a = synth[event_display]["asic"]
        ann_a = synth["ann"]["asic"]
        if ev_x.get("estimated_lcs") and ann_x.get("estimated_lcs"):
            print(
                "FPGA LC ANN/Event ratio: "
                f"{ann_x['estimated_lcs'] / ev_x['estimated_lcs']:.2f}x"
            )
        if ev_a.get("cells") and ann_a.get("cells"):
            print(
                "ASIC generic-cell ANN/Event ratio: "
                f"{ann_a['cells'] / ev_a['cells']:.2f}x"
            )


if __name__ == "__main__":
    main()
