"""
merge_results_v2.py - Merge v2 per-slice results.

The v2 comparison is:
    event SNN   : sparse event-driven SNN accelerator
    dense SNN   : dense scan SNN baseline
    INT8 ANN    : conventional quantized MLP baseline

Usage:
    python python/merge_results_v2.py
"""

import glob
import os

import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


DESIGNS = [
    ('event', 'Event SNN', '#4ECDC4'),
    ('dense', 'Dense SNN', '#FF6B6B'),
    ('ann', 'INT8 ANN', '#5567E8'),
]


def load_one(path):
    metrics = {}
    for _, row in pd.read_csv(path).iterrows():
        value = row['value']
        try:
            value = int(value)
        except (ValueError, TypeError):
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
        metrics[row['metric']] = value
    return metrics


def merge_design(sim_dir, design):
    files = sorted(glob.glob(os.path.join(sim_dir, f'metrics_classify_{design}_*.csv')))
    if not files:
        single = os.path.join(sim_dir, f'metrics_classify_{design}.csv')
        files = [single] if os.path.exists(single) else []
    if not files:
        return None

    total = {
        'num_images': 0,
        'correct': 0,
        'active_cycles': 0,
        'synapse_ops': 0,
        'mac_ops': 0,
        'spikes_fired': 0,
    }
    for path in files:
        metrics = load_one(path)
        for key in total:
            total[key] += int(metrics.get(key, 0))
    total['n_slices'] = len(files)
    total['accuracy_pct'] = 100.0 * total['correct'] / max(total['num_images'], 1)
    return total


def bar(ax, labels, values, colors, title, ylabel, fmt='{:,}'):
    bars = ax.bar(labels, values, color=colors, width=0.58,
                  edgecolor='white', linewidth=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    top = max(values) if max(values) > 0 else 1
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + top * 0.02,
                fmt.format(value), ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylim(0, top * 1.22)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, axis='y', alpha=0.3)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(script_dir, '..', 'sim')
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    labels = []
    colors = []
    for design, label, color in DESIGNS:
        merged = merge_design(sim_dir, design)
        if merged is None:
            print(f"  [WARN] No metrics files found for {design}")
            continue
        rows.append({'design': design, 'label': label, **merged})
        labels.append(label)
        colors.append(color)

    if len(rows) < 2:
        print("  [INFO] Not enough result files found in sim/. Run the v2 sbatch job first.")
        return

    print("=" * 72)
    print("  MNIST v2 benchmark - Event SNN vs Dense SNN vs INT8 ANN")
    print("=" * 72)
    for row in rows:
        ops = row['mac_ops'] if row['design'] == 'ann' else row['synapse_ops']
        print(f"  {row['label']:<10}  images={row['num_images']:>3}  "
              f"acc={row['accuracy_pct']:>6.2f}%  "
              f"active={row['active_cycles']:>14,}  ops={ops:>14,}")
    print("=" * 72)

    out_csv = os.path.join(results_dir, 'mnist_v2_comparison.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    op_values = [r['mac_ops'] if r['design'] == 'ann' else r['synapse_ops'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    bar(axes[0], labels, [r['accuracy_pct'] for r in rows], colors,
        'Classification Accuracy', 'Accuracy (%)', fmt='{:.2f}%')
    axes[0].set_ylim(0, 109)
    bar(axes[1], labels, op_values, colors, 'Operations', 'Synapse ops / MACs')
    bar(axes[2], labels, [r['active_cycles'] for r in rows], colors,
        'Active Cycles', 'Cycles')
    fig.suptitle('MNIST v2 Accelerator Comparison', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    out_png = os.path.join(results_dir, 'mnist_v2_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_png}")


if __name__ == '__main__':
    main()

