"""Quantize a model's weights to 8/4/2 bits and measure the loss on the full validation sets.

Fake quantization: each chosen weight is rounded to `bits` bits and right away dequantized back
to float, so the model runs unchanged and only the rounding error is left. Packed files come later.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from model import Config, Arke0

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="arke0_web_last", help="checkpoint <name>.pt to quantize")
parser.add_argument("--bits", type=int, nargs="+", default=[16, 8, 4, 2], help="16 = not quantized (the reference)")
parser.add_argument("--group", type=int, default=64, help="consecutive weights that share one scale")
parser.add_argument("--embedding", action="store_true",
                    help="also quantize the token embedding (it is the output layer too: weight tying)")
parser.add_argument("--scheme", choices=["sym", "asym"], default="sym",
                    help="sym: one scale per group (absmax); asym: a scale and a minimum per group (min-max)")
args = parser.parse_args()
device = "cuda"

def quantize(w, bits):
    """Symmetric (absmax): round every row of w to integers in [-qmax, qmax], one scale per row."""
    qmax = 2 ** (bits - 1) - 1                                                 # 8 bit: 127, 4 bit: 7, 2 bit: 1
    scale = (w.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)       # clamp: an all-zero row gives 0/0
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

def quantize_asym(w, bits):
    """Asymmetric (min-max): spread every row's [min, max] over the integers 0 .. 2**bits - 1."""
    qmax = 2 ** bits - 1                                                       # 8 bit: 255, 4 bit: 15, 2 bit: 3
    low = w.amin(dim=-1, keepdim=True)
    scale = ((w.amax(dim=-1, keepdim=True) - low) / qmax).clamp(min=1e-12)     # clamp: a row of equal values gives 0/0
    q = torch.round((w - low) / scale).clamp(0, qmax).to(torch.uint8)
    return q, scale, low

def dequantize_asym(q, scale, low):
    return q.float() * scale + low

def fake_quantize(w, bits, group, scheme):
    """Quantize w in groups of `group` consecutive weights; return the dequantized copy."""
    G = w.float().reshape(-1, group)                                           # each row is one group
    if scheme == "sym":
        G_hat = dequantize(*quantize(G, bits))
    else:
        G_hat = dequantize_asym(*quantize_asym(G, bits))
    return G_hat.reshape(w.shape).to(w.dtype)

ckpt = torch.load(ROOT / f"{args.name}.pt", map_location="cpu", weights_only=False)
cfg = Config(**ckpt["config"])
model = Arke0(cfg).to(device)
model.eval()

# named_parameters lists the tied embedding once, as tok_emb.weight (head.weight is the same tensor)
params = dict(model.named_parameters())
chosen = [n for n, p in params.items() if n.startswith("blocks.") and p.dim() == 2]   # Linear weights
if args.embedding:
    chosen.append("tok_emb.weight")
n_total = sum(p.numel() for p in params.values())
n_quant = sum(params[n].numel() for n in chosen)
print(f"{args.name}: {n_total / 1e6:.2f}M parameters, quantizing {n_quant / 1e6:.2f}M "
      f"({n_quant / n_total:.1%}) in groups of {args.group}, scheme {args.scheme}, the rest stays bf16")

meta = json.loads((ROOT / "data" / "meta.json").read_text(encoding="utf-8"))
chars_per_token = {s: v["chars"] / v["tokens"] for s, v in meta["val_sources"].items()}
val_sets = {s: np.memmap(ROOT / "data" / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("val", "val_tr", "val_en")}

@torch.no_grad()
def full_loss(data, batch=256):
    """Mean loss over every non-overlapping block_size window of the file: no sampling noise."""
    T = cfg.block_size
    n = (len(data) - 1) // T
    total = 0.0
    for b in range(0, n, batch):
        starts = np.arange(b, min(b + batch, n)) * T
        buf = torch.from_numpy(np.stack([data[i:i + T + 1] for i in starts]).astype(np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(buf[:, :-1].contiguous(), buf[:, 1:].contiguous())
        total += loss.item() * len(starts)
    return total / n

ref = None
for bits in args.bits:
    t0 = time.time()
    model.load_state_dict(ckpt["model"])                      # start from the original weights every time
    if bits < 16:
        with torch.no_grad():
            for n in chosen:
                params[n].copy_(fake_quantize(params[n], bits, args.group, args.scheme))
        bits_per_weight = bits + (16 if args.scheme == "sym" else 32) / args.group   # fp16 scale (+ fp16 minimum) per group
    else:
        bits_per_weight = 16
    size_mb = (n_quant * bits_per_weight + (n_total - n_quant) * 16) / 8 / 1e6
    loss = {s: full_loss(d) for s, d in val_sets.items()}
    ref = ref if ref is not None else loss["val"]
    print(f"{bits:2} bit | {bits_per_weight:5.2f} bit/weight | {size_mb:6.1f} MB | val {loss['val']:.4f} "
          f"({loss['val'] - ref:+.4f}) | tr {loss['val_tr'] / chars_per_token['tr']:.4f}/ch "
          f"| en {loss['val_en'] / chars_per_token['en']:.4f}/ch | {time.time() - t0:.0f}s")