import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"

files = sorted(RUNS.glob("*.json")) if RUNS.exists() else []
if not files:
    print(f"No run logs in {RUNS}. Train first, e.g. train.py --name xs --n_layer 4 --n_embd 256 --n_head 4")
    sys.exit(1)

runs = [json.loads(f.read_text()) for f in files]
runs.sort(key=lambda r: r["params_non_emb"])

print(f"{'run':<10} {'params':>8} {'non-emb':>8} {'train':>7} {'val':>7} {'gap':>6} {'time':>7}")
for r in runs:
    last = r["history"][-1]
    print(f"{r['name']:<10} {r['params'] / 1e6:7.2f}M {r['params_non_emb'] / 1e6:7.2f}M "
          f"{last['train']:7.3f} {last['val']:7.3f} {last['val'] - last['train']:6.3f} "
          f"{r['train_time'] / 60:5.1f}min")

fig, (ax_size, ax_curve) = plt.subplots(1, 2, figsize=(12, 4.5))

# left: final loss vs model size; the train/val gap shows overfitting
params = [r["params_non_emb"] for r in runs]
ax_size.plot(params, [r["history"][-1]["val"] for r in runs], "o-", label="val")
ax_size.plot(params, [r["history"][-1]["train"] for r in runs], "o--", label="train")
for r in runs:
    ax_size.annotate(r["name"], (r["params_non_emb"], r["history"][-1]["val"]),
                     textcoords="offset points", xytext=(5, 5))
ax_size.set_xscale("log")
ax_size.minorticks_off()                     # label only the runs, e.g. "10.6M"
ax_size.set_xticks(params, [f"{p / 1e6:.1f}M" for p in params])
ax_size.set_xlabel("parameters (without embeddings)")
ax_size.set_ylabel("final loss / token")
ax_size.set_title("Loss vs model size")
ax_size.legend()
ax_size.grid(True, alpha=0.3)

# right: training curves, val solid and train dashed in the same color
for r in runs:
    steps = [h["step"] for h in r["history"]]
    line, = ax_curve.plot(steps, [h["val"] for h in r["history"]], label=r["name"])
    ax_curve.plot(steps, [h["train"] for h in r["history"]], "--", color=line.get_color())
ax_curve.set_xlabel("step")
ax_curve.set_ylabel("loss / token (solid val, dashed train)")
ax_curve.set_title("Training curves")
ax_curve.legend()
ax_curve.grid(True, alpha=0.3)

fig.tight_layout()
out = RUNS / "scaling.png"
fig.savefig(out, dpi=120)
print("saved:", out.relative_to(ROOT))
