"""Plot the first held-out condition of each fixed-command study."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

folder = Path("artifacts/fixed_policy_loads/held")
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for mode, color, style, label in (("A", "#313842", "-", "A   Ordinary"),
                                  ("B", "#ba5311", "--", "B   Force input"),
                                  ("C", "#197080", ":", "C   Matched target")):
    for ax, task, force, key in zip(axes.flat, ("box", "tether", "standing_payload", "pulse"),
                                    (12, -16, 2, 2), ("box_progress", "box_progress", "finger", "z")):
        suffix = mode if task == "pulse" else f"{mode}{0 if mode == 'A' else force:+g}"
        with np.load(folder/f"{task}_100_{suffix}.npz") as d:
            y = d[key]
            if task == "standing_payload":
                y = y[:, 2] - d["free_nominal"]
            if task == "pulse":
                y -= d["unperturbed"]
            ax.plot(.02*(1+np.arange(len(y))), 100*y, style, color=color, label=label, lw=2)
for ax, name, units in zip(axes.flat,
                           ("Box push  ·  +12 N forward", "Tether pull  ·  −16 N along walk",
                            "Standing payload  ·  +2 N upward", "Downward physical pulse  ·  +2 N upward"),
                           ("Box progress (cm)", "Box progress (cm)",
                            "Height vs. unloaded A (cm)", "Height vs. own pulse-free twin (cm)")):
    ax.set_title(name, loc="left", fontsize=12)
    ax.set_ylabel(units)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 3)
axes[1, 1].axvspan(.8, 1.2, color=".4", alpha=.1, zorder=-2)
for ax in axes[-1]:
    ax.set_xlabel("Time after common snapshot (s)")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.suptitle("Fixed commands under physical load   |   held-out condition 100",
             x=.06, ha="left", fontsize=15)
fig.tight_layout(rect=(0, .06, 1, .96))
fig.savefig("docs/fixed_policy_loads.png", dpi=160)
