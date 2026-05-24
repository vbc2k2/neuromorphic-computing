#!/usr/bin/env python3
"""
replay_event_python.py - Replay exported event SNN files with RTL-style ordering.

This is a debugging model, not a trainer. It reads sim/*.mem and sim/snn_config.vh,
then replays the event FIFO, CSR deliveries, neuron-state scan, output membrane
readout, and signed membrane wrapping in the same high-level order as
top_event_ram/top_event_ram2. Use it to decide whether a Python-vs-RTL mismatch
comes from the compact exporter model or from the RTL implementation.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Deque


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"


def parse_config(path: Path) -> dict[str, str]:
    cfg: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) >= 3 and parts[0] == "`define":
            value = parts[2].strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            cfg[parts[1]] = value
    return cfg


def cfg_int(cfg: dict[str, str], key: str, fallback: int = 0) -> int:
    try:
        return int(cfg.get(key, fallback))
    except ValueError:
        return fallback


def read_hex_values(path: Path, signed_width: int | None = None) -> list[int]:
    values: list[int] = []
    sign = 1 << (signed_width - 1) if signed_width else 0
    mask = (1 << signed_width) - 1 if signed_width else 0
    with path.open() as f:
        for token in f.read().split():
            value = int(token, 16)
            if signed_width is not None:
                value &= mask
                if value & sign:
                    value -= 1 << signed_width
            values.append(value)
    return values


def read_labels(path: Path) -> list[int]:
    return [int(token) for token in path.read_text().split()]


def read_spike_words(path: Path) -> list[str]:
    return path.read_text().split()


def wrap_signed(value: int, width: int) -> int:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    return ((value & mask) ^ sign) - sign


def leak_value(value: int, leak: int, width: int) -> int:
    if value > leak:
        return wrap_signed(value - leak, width)
    if value > 0:
        return 0
    return wrap_signed(value, width)


def argmax_first(values: list[int]) -> int:
    best_idx = 0
    best = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best:
            best_idx = idx
            best = value
    return best_idx


def spike_ids(word: str, n_input: int) -> list[int]:
    # Export writes bit N_INPUT-1 at the left and bit 0 at the right.
    return [idx for idx in range(n_input) if word[n_input - 1 - idx] == "1"]


class EventReplay:
    def __init__(self, sim_dir: Path):
        cfg = parse_config(sim_dir / "snn_config.vh")
        self.n_total = cfg_int(cfg, "SNN_N_TOTAL")
        self.n_input = cfg_int(cfg, "SNN_N_INPUT")
        self.n_output = cfg_int(cfg, "SNN_N_OUTPUT")
        self.id_bias = cfg_int(cfg, "SNN_ID_BIAS")
        self.id_output_base = cfg_int(cfg, "SNN_ID_OUTPUT_BASE")
        self.t_steps = cfg_int(cfg, "SNN_T_STEPS")
        self.threshold = cfg_int(cfg, "SNN_THRESHOLD")
        self.leak = cfg_int(cfg, "SNN_LEAK", 1)
        self.mem_width = cfg_int(cfg, "SNN_MEMBRANE_WIDTH", 16)

        self.csr_dst = read_hex_values(sim_dir / "csr_dst.mem")
        self.csr_weight = read_hex_values(sim_dir / "csr_weight.mem", signed_width=8)
        self.src_start = read_hex_values(sim_dir / "csr_start.mem")
        self.src_count = read_hex_values(sim_dir / "csr_count.mem")
        self.spike_words = read_spike_words(sim_dir / "snn_spikes.mem")
        self.labels = read_labels(sim_dir / "snn_labels.mem")

    def is_output(self, neuron_id: int) -> bool:
        return self.id_output_base <= neuron_id < self.id_output_base + self.n_output

    def deliver_source(self, src: int, weight_sum: list[int]) -> int:
        deliveries = 0
        start = self.src_start[src]
        count = self.src_count[src]
        for idx in range(start, start + count):
            dst = self.csr_dst[idx]
            weight_sum[dst] = wrap_signed(weight_sum[dst] + self.csr_weight[idx], self.mem_width)
            deliveries += 1
        return deliveries

    def run_image(self, image: int, trace: bool = False) -> tuple[int, list[int], dict[str, int], list[list[int]]]:
        membrane = [0] * self.n_total
        weight_sum = [0] * self.n_total
        fifo: Deque[int] = deque()
        stats = {"deliveries": 0, "router_events": 0, "spikes": 0}
        trace_rows: list[list[int]] = []

        for t in range(self.t_steps):
            word = self.spike_words[image * self.t_steps + t]
            for src in spike_ids(word, self.n_input):
                fifo.append(src)
            fifo.append(self.id_bias)

            while fifo:
                src = fifo.popleft()
                stats["router_events"] += 1
                stats["deliveries"] += self.deliver_source(src, weight_sum)

            next_fifo: Deque[int] = deque()
            for neuron_id in range(self.n_total):
                leaked = leak_value(membrane[neuron_id], self.leak, self.mem_width)
                integrated = wrap_signed(leaked + weight_sum[neuron_id], self.mem_width)
                fired = integrated >= self.threshold
                if fired and not self.is_output(neuron_id):
                    membrane[neuron_id] = 0
                    next_fifo.append(neuron_id)
                    stats["spikes"] += 1
                else:
                    membrane[neuron_id] = integrated
                weight_sum[neuron_id] = 0

            fifo = next_fifo
            if trace:
                trace_rows.append([
                    t,
                    *membrane[self.id_output_base:self.id_output_base + self.n_output],
                ])

        scores = membrane[self.id_output_base:self.id_output_base + self.n_output]
        return argmax_first(scores), scores, stats, trace_rows


def write_trace(path: Path, trace_rows: list[list[int]], n_output: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", *[f"score{i}" for i in range(n_output)]])
        writer.writerows(trace_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay exported event SNN files in Python")
    parser.add_argument("--sim-dir", type=Path, default=SIM)
    parser.add_argument("--tag", default="_replay")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int)
    parser.add_argument("--trace-image", type=int)
    args = parser.parse_args()

    replay = EventReplay(args.sim_dir)
    last = args.last if args.last is not None else len(replay.labels) - 1
    last = min(last, len(replay.labels) - 1)
    if args.first < 0 or args.first > last:
        raise SystemExit("invalid image range")

    out_path = args.sim_dir / f"classify_event_python_replay{args.tag}.csv"
    correct = 0
    total_deliveries = 0
    total_events = 0
    total_spikes = 0
    nrun = last - args.first + 1

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "label", "prediction", "correct",
            "deliveries", "router_events", "spikes",
            *[f"score{i}" for i in range(replay.n_output)],
        ])
        for image in range(args.first, last + 1):
            pred, scores, stats, trace_rows = replay.run_image(
                image, trace=(args.trace_image == image)
            )
            ok = int(pred == replay.labels[image])
            correct += ok
            total_deliveries += stats["deliveries"]
            total_events += stats["router_events"]
            total_spikes += stats["spikes"]
            writer.writerow([
                image, replay.labels[image], pred, ok,
                stats["deliveries"], stats["router_events"], stats["spikes"],
                *scores,
            ])
            if trace_rows:
                trace_path = args.sim_dir / f"trace_event_python_replay{args.tag}_img{image}.csv"
                write_trace(trace_path, trace_rows, replay.n_output)
                print(f"wrote trace: {trace_path}")

    print("=" * 78)
    print(f"Python RTL-style event replay tag={args.tag}")
    print(f"images {args.first}..{last}")
    print(f"accuracy: {correct}/{nrun} = {100.0 * correct / max(nrun, 1):.2f}%")
    print(f"deliveries: {total_deliveries}")
    print(f"router events: {total_events}")
    print(f"spikes: {total_spikes}")
    print(f"wrote: {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
