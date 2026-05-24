#!/usr/bin/env python3
"""compare_trace.py - Compare per-timestep score traces."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_trace(path: Path) -> list[list[int]]:
    if not path.exists():
        raise SystemExit(f"missing trace: {path}")
    rows: list[list[int]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            scores = [int(value) for key, value in row.items() if key.startswith("score")]
            rows.append(scores)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare score traces")
    parser.add_argument("golden", type=Path)
    parser.add_argument("rtl", type=Path)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    golden = read_trace(args.golden)
    rtl = read_trace(args.rtl)
    n = min(len(golden), len(rtl))
    if n == 0:
        raise SystemExit("no trace rows to compare")

    mismatch_count = 0
    max_abs_diff = 0
    first_mismatches: list[tuple[int, list[int], list[int], int]] = []
    for t in range(n):
        diffs = [abs(a - b) for a, b in zip(golden[t], rtl[t])]
        row_max = max(diffs, default=0)
        max_abs_diff = max(max_abs_diff, row_max)
        if row_max:
            mismatch_count += 1
            if len(first_mismatches) < args.show:
                first_mismatches.append((t, golden[t], rtl[t], row_max))

    print("=" * 78)
    print("Trace compare")
    print(f"golden: {args.golden}")
    print(f"rtl:    {args.rtl}")
    print("=" * 78)
    print(f"timesteps compared: {n}")
    print(f"mismatch timesteps: {mismatch_count}")
    print(f"max abs diff:       {max_abs_diff}")
    if first_mismatches:
        print()
        print("first mismatches:")
        for t, golden_scores, rtl_scores, row_max in first_mismatches:
            print(f"t={t} max_abs_diff={row_max}")
            print(f"  golden {golden_scores}")
            print(f"  rtl    {rtl_scores}")


if __name__ == "__main__":
    main()
