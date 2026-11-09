"""Plot a model size experiment from the run logs in runs/.

Compare runs that saw the same data and the same number of tokens, so that only the size differs.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # one per run, smallest model first
INK, MUTED = "#0b0b0b", "#52514e"

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", default=["v2_xs", "v2_s", "v2_m", "arke0_clean"],
                    help="run logs runs/<name>.json to compare")
parser.add_argument("--out", default="docs/scaling.png", help="where to write the figure")
args = parser.parse_args()
if len(args.runs) > len(COLORS):
    parser.error(f"at most {len(COLORS)} runs: more lines than that cannot be told apart")

files = [RUNS / f"{name}.json" for name in args.runs]
missing = [f.name for f in files if not f.exists()]
if missing:
    sys.exit(f"No run log for {', '.join(missing)} in {RUNS}. Train first, e.g. train.py --name xs --n_layer 4 --n_embd 256 --n_head 4")

runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
runs.sort(key=lambda r: r["params"])

print(f"{'run':<12} {'params':>8} {'non-emb':>8} {'train':>7} {'val':>7} {'gap':>6} {'time':>9}")
for r in runs:
    last = r["history"][-1]
    print(f"{r['name']:<12} {r['params'] / 1e6:7.2f}M {r['params_non_emb'] / 1e6:7.2f}M "
          f"{last['train']:7.3f} {last['val']:7.3f} {last['val'] - last['train']:6.3f} "
          f"{r['train_time'] / 60:7.1f}min")

fig, (ax_size, ax_curve) = plt.subplots(1, 2, figsize=(12, 4.5))
final_val = [r["history"][-1]["val"] for r in runs]
final_train = [r["history"][-1]["train"] for r in runs]

# left: final loss vs model size; the train/val gap shows overfitting. Color names a run on both panels,
# so the lines that join the runs stay neutral.
params = [r["params"] for r in runs]
ax_size.plot(params, final_val, "-", color=MUTED, linewidth=2, label="val", zorder=1)
ax_size.plot(params, final_train, "--", color=MUTED, linewidth=2, label="train", zorder=1)
for r, p, val, train, color in zip(runs, params, final_val, final_train, COLORS):
    ax_size.scatter([p, p], [val, train], s=64, color=color, edgecolors="white", linewidths=1.5, zorder=2)
    ax_size.annotate(f"{r['name']}\n{val:.3f}", (p, val), textcoords="offset points", xytext=(8, 8), color=INK, fontsize=9)
ax_size.set_xscale("log")
ax_size.minorticks_off()                     # label only the runs, e.g. "17.0M"
ax_size.set_xticks(params, [f"{p / 1e6:.1f}M" for p in params])
ax_size.set_xlabel("parameters")
ax_size.set_ylabel("final loss / token")
ax_size.set_title("Loss vs model size")
ax_size.set_xlim(min(params) * 0.8, max(params) * 1.6)   # room for the last label
ax_size.set_ylim(top=max(final_val) + 0.12)
ax_size.legend(frameon=False)

# right: training curves, val solid and train dashed in the same color
for r, color in zip(runs, COLORS):
    steps = [h["step"] for h in r["history"]]
    ax_curve.plot(steps, [h["val"] for h in r["history"]], color=color, linewidth=2, label=r["name"])
    ax_curve.plot(steps, [h["train"] for h in r["history"]], "--", color=color, linewidth=1)
ax_curve.set_xlabel("step")
ax_curve.set_ylabel("loss / token (solid val, dashed train)")
ax_curve.set_title("Training curves")
ax_curve.set_ylim(min(final_train) - 0.05, max(final_val) + 0.5)   # the first steps (loss ~10) would flatten the rest
ax_curve.legend(frameon=False)

for ax in (ax_size, ax_curve):
    ax.grid(True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=120)
print("saved:", out.relative_to(ROOT))
