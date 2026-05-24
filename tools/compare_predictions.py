#!/usr/bin/env python3
"""
compare_predictions.py - Compare exported Python SNN predictions to RTL output.

The exporter writes sim/snn_golden_predictions.csv for the current sim/*.mem
files. Verilator writes sim/classify_event<TAG>.csv. This script makes the
Python-vs-RTL disagreement visible before we trust a tuning point.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "sim"


def read_rows(path: Path) -> dict[int, dict[str, int]]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    rows: dict[int, dict[str, int]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            image = int(row["image"])
            rows[image] = {
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "correct": int(row["correct"]),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Python golden SNN predictions with RTL predictions"
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--golden", type=Path, default=SIM / "snn_golden_predictions.csv")
    parser.add_argument("--rtl", type=Path)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    rtl_path = args.rtl or (SIM / f"classify_event{args.tag}.csv")
    golden = read_rows(args.golden)
    rtl = read_rows(rtl_path)

    common = sorted(set(golden) & set(rtl))
    if not common:
        raise SystemExit("no overlapping image ids")

    same_pred = 0
    golden_correct = 0
    rtl_correct = 0
    rtl_lost = 0
    rtl_gained = 0
    mismatches: list[tuple[int, int, int, int, int, int]] = []

    for image in common:
        g = golden[image]
        r = rtl[image]
        if g["label"] != r["label"]:
            raise SystemExit(
                f"label mismatch on image {image}: golden={g['label']} rtl={r['label']}"
            )
        same = int(g["prediction"] == r["prediction"])
        same_pred += same
        golden_correct += g["correct"]
        rtl_correct += r["correct"]
        if g["correct"] and not r["correct"]:
            rtl_lost += 1
        if r["correct"] and not g["correct"]:
            rtl_gained += 1
        if not same:
            mismatches.append((
                image,
                g["label"],
                g["prediction"],
                r["prediction"],
                g["correct"],
                r["correct"],
            ))

    n = len(common)
    print("=" * 78)
    print(f"Prediction compare tag={args.tag}")
    print(f"golden: {args.golden}")
    print(f"rtl:    {rtl_path}")
    print("=" * 78)
    print(f"images compared:       {n}")
    print(f"python golden accuracy:{100.0 * golden_correct / n:7.2f}% ({golden_correct}/{n})")
    print(f"rtl accuracy:          {100.0 * rtl_correct / n:7.2f}% ({rtl_correct}/{n})")
    print(f"prediction agreement:  {100.0 * same_pred / n:7.2f}% ({same_pred}/{n})")
    print(f"rtl lost golden-correct images:   {rtl_lost}")
    print(f"rtl gained golden-wrong images:   {rtl_gained}")

    if mismatches:
        print()
        print("first mismatches:")
        print("image  label  golden_pred  rtl_pred  golden_ok  rtl_ok")
        for image, label, gpred, rpred, gok, rok in mismatches[:args.show]:
            print(f"{image:5d}  {label:5d}  {gpred:11d}  {rpred:8d}  {gok:9d}  {rok:6d}")
    else:
        print("predictions match exactly")

    if args.fail_on_mismatch and mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
