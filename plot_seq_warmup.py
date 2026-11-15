"""Plot the sequence length warmup experiment from the run logs in runs/.

Two runs with the same recipe and token budget, one of them starting on half-length windows.
Left: val loss over training. Right: the gap between them after the switch.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
BASE_COLOR, WARM_COLOR = "#2a78d6", "#eb6834"   # same first two colors as plot_scaling.py
INK, MUTED, GRID, SHADE = "#0b0b0b", "#52514e", "#e6e5e0", "#f1f0ec"

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="v2_xs_512", help="run log without warmup")
parser.add_argument("--warm", default="v2_xs_512w", help="run log with --seq-warmup")
parser.add_argument("--out", default="docs/seq_warmup.png", help="where to write the figure")
args = parser.parse_args()

files = [RUNS / f"{name}.json" for name in (args.base, args.warm)]
missing = [f.name for f in files if not f.exists()]
if missing:
    sys.exit(f"No run log for {', '.join(missing)} in {RUNS}")
base, warm = (json.loads(f.read_text(encoding="utf-8")) for f in files)
switch = warm.get("seq_warmup", 0)
if not switch:
    sys.exit(f"{args.warm} has no seq_warmup in its run log")

b = {h["step"]: h["val"] for h in base["history"]}
w = {h["step"]: h["val"] for h in warm["history"]}
steps = sorted(set(b) & set(w))
after = [s for s in steps if s > switch]
gap = [w[s] - b[s] for s in after]
print(f"switch at {switch}; first eval after it: {after[0]} gap {gap[0]:+.4f}; final {after[-1]} gap {gap[-1]:+.4f}")

fig, (ax_loss, ax_gap) = plt.subplots(1, 2, figsize=(12, 4.5))

# left: both curves; the warmup steps are shaded, where the warm run is scored on 512 windows
# with positions it has not trained yet
low = min(w.values()) - 0.1                                    # room under the curves for the labels
ax_loss.axvspan(0, switch, color=SHADE, zorder=0)
ax_loss.text(switch / 2, low + 0.25, f"warmup\n{2 * warm['batch_size']}\n× {warm['config']['block_size'] // 2}",
             ha="center", va="center", color=MUTED, fontsize=9)
for run, vals, color, label in ((base, b, BASE_COLOR, "no warmup"), (warm, w, WARM_COLOR, f"warmup {switch:,} steps")):
    ax_loss.plot(steps, [vals[s] for s in steps], color=color, linewidth=2, label=label)
    ax_loss.annotate(f"{vals[steps[-1]]:.4f}", (steps[-1], vals[steps[-1]]), textcoords="offset points",
                     xytext=(6, -12 if run is warm else 4), color=INK, fontsize=9)
ax_loss.set_xlabel("step")
ax_loss.set_ylabel("val loss / token (512-token windows)")
ax_loss.set_title(f"{base['config']['block_size']}-token context, {base['params'] / 1e6:.1f}M parameters")
ax_loss.set_ylim(low, 5.0)                                     # the first steps (loss ~10) would flatten the rest
ax_loss.set_xlim(0, steps[-1] * 1.12)                          # room for the end labels
ax_loss.legend(frameon=False, loc="upper right")

# right: warm minus base after the switch; below zero means warmup is ahead
ax_gap.axvspan(0, switch, color=SHADE, zorder=0)
ax_gap.text(switch / 2, -0.002, "warmup", ha="center", va="top", color=MUTED, fontsize=9)
ax_gap.axhline(0, color=MUTED, linewidth=1)
ax_gap.plot(after, gap, color=WARM_COLOR, linewidth=2)
for s, g in ((after[0], gap[0]), (after[-1], gap[-1])):
    ax_gap.scatter([s], [g], s=64, color=WARM_COLOR, edgecolors="white", linewidths=1.5, zorder=3)
    ax_gap.annotate(f"{g:+.4f}", (s, g), textcoords="offset points", xytext=(6, -12), color=INK, fontsize=9)
ax_gap.set_xlabel("step")
ax_gap.set_ylabel("val loss: warmup − no warmup")
ax_gap.set_title(f"Gap after the switch at step {switch:,}")
ax_gap.set_ylim(min(gap) - 0.01, 0.005)
ax_gap.set_xlim(0, steps[-1] * 1.12)

for ax in (ax_loss, ax_gap):
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x / 1000:.0f}k" if x else "0")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print("saved:", out.relative_to(ROOT))
