"""
merge_results.py - Merge per-slice SNN benchmark results from an sbatch run.

After sbatch_mnist.sh finishes, sim/ contains per-slice metrics files:
    metrics_classify_event_0.csv .. _5.csv
    metrics_classify_dense_0.csv .. _5.csv
This script sums them into a single event vs dense comparison (accuracy +
synaptic-operation / active-cycle efficiency) and writes the plot + CSV.

Usage:  python merge_results.py
"""

import os
import glob
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR_EVENT = '#4ECDC4'
COLOR_DENSE = '#FF6B6B'


def load_one(path):
    m = {}
    for _, row in pd.read_csv(path).iterrows():
        v = row['value']
        try:
            v = int(v)
        except (ValueError, TypeError):
            try:
                v = float(v)
            except (ValueError, TypeError):
                pass
        m[row['metric']] = v
    return m


def merge(sim_dir, design):
    """Sum the per-slice metrics for one design."""
    files = sorted(glob.glob(os.path.join(sim_dir, f'metrics_classify_{design}_*.csv')))
    if not files:
        single = os.path.join(sim_dir, f'metrics_classify_{design}.csv')
        files = [single] if os.path.exists(single) else []
    if not files:
        return None
    total = {'num_images': 0, 'correct': 0, 'active_cycles': 0,
             'synapse_ops': 0, 'spikes_fired': 0}
    for f in files:
        m = load_one(f)
        for k in total:
            total[k] += int(m.get(k, 0))
    total['n_slices'] = len(files)
    return total


def _bar(ax, values, title, ylabel, fmt='{:,}'):
    bars = ax.bar(['Event-Driven', 'Dense Baseline'], values,
                  color=[COLOR_EVENT, COLOR_DENSE], width=0.55,
                  edgecolor='white', linewidth=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11)
    top = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + top * 0.02,
                fmt.format(val), ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    ax.set_ylim(0, top * 1.20)
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, axis='y')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.join(script_dir, '..', 'sim')
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    ev = merge(sim_dir, 'event')
    dn = merge(sim_dir, 'dense')
    if ev is None or dn is None:
        print("  [INFO] No per-slice result files found in sim/. Run the sbatch job first.")
        return

    acc_e = 100.0 * ev['correct'] / ev['num_images']
    acc_d = 100.0 * dn['correct'] / dn['num_images']

    print("=" * 64)
    print(f"  MNIST SNN benchmark - merged from {ev['n_slices']} slices")
    print("=" * 64)
    print(f"  Test images           : {ev['num_images']}")
    print(f"  Accuracy  event/dense : {acc_e:.2f}%  /  {acc_d:.2f}%")
    print(f"  Synaptic ops          : {ev['synapse_ops']:,}  vs  {dn['synapse_ops']:,}"
          f"   -> {dn['synapse_ops']/max(ev['synapse_ops'],1):.2f}x fewer")
    print(f"  Active cycles         : {ev['active_cycles']:,}  vs  {dn['active_cycles']:,}"
          f"   -> {dn['active_cycles']/max(ev['active_cycles'],1):.2f}x fewer")
    print("=" * 64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _bar(axes[0], [acc_e, acc_d], 'Classification Accuracy', 'Accuracy (%)',
         fmt='{:.2f}%')
    axes[0].set_ylim(0, 109)
    _bar(axes[1], [ev['synapse_ops'], dn['synapse_ops']],
         'Synaptic Operations', 'Synapse ops')
    _bar(axes[2], [ev['active_cycles'], dn['active_cycles']],
         'Active Cycles (work done)', 'Cycles')
    fig.suptitle('Event-Driven vs Dense SNN Accelerator - MNIST',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'mnist_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    pd.DataFrame([
        {'metric': 'accuracy_pct',  'event_driven': acc_e,             'dense_baseline': acc_d},
        {'metric': 'synapse_ops',   'event_driven': ev['synapse_ops'], 'dense_baseline': dn['synapse_ops']},
        {'metric': 'active_cycles', 'event_driven': ev['active_cycles'],'dense_baseline': dn['active_cycles']},
    ]).to_csv(os.path.join(results_dir, 'mnist_comparison.csv'), index=False)
    print(f"  Saved: {os.path.join(results_dir, 'mnist_comparison.csv')}")


if __name__ == '__main__':
    main()
