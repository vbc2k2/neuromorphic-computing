"""
compare_classify.py - Event-driven vs dense-baseline SNN digit classifier

Reads the metrics produced by tb_classify (event-driven) and tb_classify_dense
(dense baseline) and reports:
  * classification accuracy of each design  (must be equal -> equivalent)
  * synaptic operations and active cycles    (the event-driven efficiency win)

Author: Neuromorphic Accelerator Project
"""

import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR_EVENT = '#4ECDC4'
COLOR_DENSE = '#FF6B6B'


def load_metrics(path):
    if not os.path.exists(path):
        print(f"  [WARN] missing {path}")
        return None
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

    ev = load_metrics(os.path.join(sim_dir, 'metrics_classify_event.csv'))
    dn = load_metrics(os.path.join(sim_dir, 'metrics_classify_dense.csv'))
    if ev is None or dn is None:
        print("  [INFO] Run tb_classify and tb_classify_dense first.")
        return

    acc_e = float(ev['accuracy_pct'])
    acc_d = float(dn['accuracy_pct'])
    sop_e = int(ev['synapse_ops'])
    sop_d = int(dn['synapse_ops'])
    cyc_e = int(ev['active_cycles'])
    cyc_d = int(dn['active_cycles'])

    print("=" * 64)
    print("  Event-Driven vs Dense Baseline - SNN Digit Classifier")
    print("=" * 64)
    print(f"  Test images           : {ev['num_images']}")
    print(f"  Accuracy  event/dense : {acc_e:.2f}%  /  {acc_d:.2f}%   "
          f"{'(equal -> equivalent)' if abs(acc_e-acc_d) < 1e-6 else 'MISMATCH!'}")
    print(f"  Synaptic ops          : {sop_e:,}  vs  {sop_d:,}   "
          f"-> {sop_d/sop_e:.2f}x fewer")
    print(f"  Active cycles         : {cyc_e:,}  vs  {cyc_d:,}   "
          f"-> {cyc_d/cyc_e:.2f}x fewer")
    print("=" * 64)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    _bar(axes[0], [acc_e, acc_d], 'Classification Accuracy (must be equal)',
         'Accuracy (%)', fmt='{:.2f}%')
    axes[0].set_ylim(0, 109)
    if abs(acc_e - acc_d) < 1e-6:
        axes[0].text(0.5, 0.93, 'IDENTICAL -> functionally equivalent',
                     transform=axes[0].transAxes, ha='center',
                     fontsize=10, fontweight='bold', color='#2d7d46')

    _bar(axes[1], [sop_e, sop_d], 'Synaptic Operations', 'Synapse ops')
    axes[1].text(0.5, 0.93, f'{sop_d/sop_e:.1f}x fewer',
                 transform=axes[1].transAxes, ha='center',
                 fontsize=11, fontweight='bold', color='#2d7d46')

    _bar(axes[2], [cyc_e, cyc_d], 'Active Cycles (work done)', 'Cycles')
    axes[2].text(0.5, 0.93, f'{cyc_d/cyc_e:.1f}x fewer',
                 transform=axes[2].transAxes, ha='center',
                 fontsize=11, fontweight='bold', color='#2d7d46')

    fig.suptitle('Event-Driven vs Dense SNN Accelerator - 8x8 Digit Classification',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(results_dir, 'classify_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # summary CSV
    pd.DataFrame([
        {'metric': 'accuracy_pct',  'event_driven': acc_e, 'dense_baseline': acc_d},
        {'metric': 'synapse_ops',   'event_driven': sop_e, 'dense_baseline': sop_d},
        {'metric': 'active_cycles', 'event_driven': cyc_e, 'dense_baseline': cyc_d},
    ]).to_csv(os.path.join(results_dir, 'classify_comparison.csv'), index=False)
    print(f"  Saved: {os.path.join(results_dir, 'classify_comparison.csv')}")


if __name__ == '__main__':
    main()
