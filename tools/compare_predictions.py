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


def read_rows(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"missing file: {path}")
    rows: dict[int, dict[str, object]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        score_cols = sorted(
            [name for name in (reader.fieldnames or []) if name.startswith("score")],
            key=lambda name: int(name[5:]),
        )
        for row in reader:
            image = int(row["image"])
            rows[image] = {
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "correct": int(row["correct"]),
                "scores": [int(row[name]) for name in score_cols],
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
    parser.add_argument("--show-scores", type=int, default=5)
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
    score_rows = 0
    score_mismatch_rows = 0
    max_abs_score_diff = 0
    sum_max_abs_score_diff = 0
    mismatches: list[tuple[int, int, int, int, int, int]] = []
    score_mismatches: list[tuple[int, list[int], list[int]]] = []

    for image in common:
        g = golden[image]
        r = rtl[image]
        if g["label"] != r["label"]:
            raise SystemExit(
                f"label mismatch on image {image}: golden={g['label']} rtl={r['label']}"
            )
        gpred = int(g["prediction"])
        rpred = int(r["prediction"])
        gok = int(g["correct"])
        rok = int(r["correct"])
        same = int(gpred == rpred)
        same_pred += same
        golden_correct += gok
        rtl_correct += rok
        if gok and not rok:
            rtl_lost += 1
        if rok and not gok:
            rtl_gained += 1
        if not same:
            mismatches.append((
                image,
                int(g["label"]),
                gpred,
                rpred,
                gok,
                rok,
            ))
        gscores = g.get("scores") or []
        rscores = r.get("scores") or []
        if gscores and rscores and len(gscores) == len(rscores):
            score_rows += 1
            diffs = [abs(int(a) - int(b)) for a, b in zip(gscores, rscores)]
            row_max = max(diffs, default=0)
            sum_max_abs_score_diff += row_max
            max_abs_score_diff = max(max_abs_score_diff, row_max)
            if row_max != 0:
                score_mismatch_rows += 1
                if len(score_mismatches) < args.show_scores:
                    score_mismatches.append((image, list(map(int, gscores)), list(map(int, rscores))))

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
    if score_rows:
        print(f"score rows compared:    {score_rows}")
        print(f"score mismatch rows:    {score_mismatch_rows}")
        print(f"max abs score diff:     {max_abs_score_diff}")
        print(f"avg max abs score diff: {sum_max_abs_score_diff / score_rows:.2f}")

    if mismatches:
        print()
        print("first mismatches:")
        print("image  label  golden_pred  rtl_pred  golden_ok  rtl_ok")
        for image, label, gpred, rpred, gok, rok in mismatches[:args.show]:
            print(f"{image:5d}  {label:5d}  {gpred:11d}  {rpred:8d}  {gok:9d}  {rok:6d}")
    else:
        print("predictions match exactly")

    if score_mismatches:
        print()
        print("first score-vector mismatches:")
        for image, gscores, rscores in score_mismatches:
            print(f"image {image}:")
            print(f"  golden {gscores}")
            print(f"  rtl    {rscores}")

    if args.fail_on_mismatch and mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
