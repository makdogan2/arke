"""Quantize a model's weights to 8/4/2 bits and measure the loss on the full validation sets.

Fake quantization: each chosen weight is rounded to `bits` bits and right away dequantized back
to float, so the model runs unchanged and only the rounding error is left.
--packed (4-bit asym) really packs the weights, two per byte with fp16 scales and minimums, and unpacks them.
--save writes the packed model to one file; --load rebuilds the model from that file alone and measures it.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
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
parser.add_argument("--sample", action="store_true",
                    help="generate text from fixed prompts and a fixed seed instead of measuring the loss")
parser.add_argument("--packed", action="store_true",
                    help="4-bit asym only: pack the weights two per byte with fp16 scales and minimums, then unpack")
parser.add_argument("--save", help="with --packed: write the packed model to this file, e.g. arke0_web_q4.pt")
parser.add_argument("--load", help="measure a file written by --save, without the original checkpoint")
args = parser.parse_args()
if args.load and (args.packed or args.save or args.embedding):
    parser.error("--load takes everything from the file: no --packed, --save or --embedding")
if args.packed and (args.scheme != "asym" or any(b not in (4, 16) for b in args.bits)):
    parser.error("--packed only does 4-bit asym: use --scheme asym --bits 16 4")
if args.save and not args.packed:
    parser.error("--save needs --packed")
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

def pack4(q):
    """q: uint8 values in 0..15, even count. Element 2i goes to the low 4 bits of byte i, 2i+1 to the high 4 bits."""
    q = q.reshape(-1)
    assert q.numel() % 2 == 0 and int(q.max()) <= 15
    return q[0::2] | (q[1::2] << 4)

def unpack4(packed):
    low = packed & 0x0F                                     # 0x0F = 0000 1111: keep the low 4 bits
    high = packed >> 4                                      # shift the high 4 bits down
    return torch.stack([low, high], dim=-1).reshape(-1)     # interleave back: low0, high0, low1, high1, ...

def pack_weight(w, group):
    """4-bit asym: everything a file needs to rebuild w. Scale and minimum are stored as fp16."""
    q, scale, low = quantize_asym(w.float().reshape(-1, group), 4)
    return {"q": pack4(q), "scale": scale.half(), "low": low.half(), "shape": tuple(w.shape)}

def unpack_weight(p, group):
    q = unpack4(p["q"]).reshape(-1, group)
    return dequantize_asym(q, p["scale"].float(), p["low"].float()).reshape(p["shape"])

def packed_bytes(packed):
    return sum(t.numel() * t.element_size() for p in packed.values() for t in (p["q"], p["scale"], p["low"]))

# weights_only=True: only tensors, dicts, lists, numbers and strings can come out of the file, never code
ckpt = torch.load(ROOT / (args.load or f"{args.name}.pt"), map_location="cpu", weights_only=True)
cfg = Config(**ckpt["config"])
model = Arke0(cfg).to(device)
model.eval()

# named_parameters lists the tied embedding once, as tok_emb.weight (head.weight is the same tensor)
params = dict(model.named_parameters())
if args.load:                                                 # the file decides what is packed and how
    chosen, args.group, args.scheme, args.bits = list(ckpt["packed"]), ckpt["group"], "asym", [4]
else:
    chosen = [n for n, p in params.items() if n.startswith("blocks.") and p.dim() == 2]   # Linear weights
    if args.embedding:
        chosen.append("tok_emb.weight")
n_total = sum(p.numel() for p in params.values())
n_quant = sum(params[n].numel() for n in chosen)
source = f"{args.load} ({(ROOT / args.load).stat().st_size / 1e6:.1f} MB on disk)" if args.load else args.name
print(f"{source}: {n_total / 1e6:.2f}M parameters, quantizing {n_quant / 1e6:.2f}M "
      f"({n_quant / n_total:.1%}) in groups of {args.group}, scheme {args.scheme}, the rest stays bf16")

def load_packed(f):
    """Fill the model from a --save file alone: the bf16 rest as it is, the 4-bit weights unpacked."""
    missing = set(params) - set(f["rest"]) - set(f["packed"])
    assert not missing, f"the file has no values for {sorted(missing)}"   # they would silently stay random
    with torch.no_grad():
        for n, t in f["rest"].items():
            params[n].copy_(t)                                # bf16 -> the model's fp32
        for n, p in f["packed"].items():
            params[n].copy_(unpack_weight(p, f["group"]))

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

tok = Tokenizer.from_file(str(ROOT / "data" / ckpt["tokenizer"]))
PROMPTS = ["Türkiye, ", "Bilim insanları ", "The history of ", "Scientists have "]

@torch.no_grad()
def sample(prompt, n=100, seed=1337):
    """Same seed for every bit width: the random draws match, so only the weights differ."""
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature=0.8, top_k=50)[0].tolist()
    return tok.decode(out, skip_special_tokens=False).replace("<|endoftext|>", "\n\n<|endoftext|>\n\n")

ref = None
for bits in args.bits:
    t0 = time.time()
    if not args.load:
        model.load_state_dict(ckpt["model"])                  # start from the original weights every time
    if args.load:
        load_packed(ckpt)
        bits_per_weight = packed_bytes(ckpt["packed"]) * 8 / n_quant
    elif bits < 16 and args.packed:
        packed = {n: pack_weight(params[n], args.group) for n in chosen}
        with torch.no_grad():
            for n in chosen:
                params[n].copy_(unpack_weight(packed[n], args.group))
        bits_per_weight = packed_bytes(packed) * 8 / n_quant     # measured from the packed tensors, not computed
    elif bits < 16:
        with torch.no_grad():
            for n in chosen:
                params[n].copy_(fake_quantize(params[n], bits, args.group, args.scheme))
        bits_per_weight = bits + (16 if args.scheme == "sym" else 32) / args.group   # fp16 scale (+ fp16 minimum) per group
    else:
        bits_per_weight = 16
    size_mb = (n_quant * bits_per_weight + (n_total - n_quant) * 16) / 8 / 1e6
    if args.save and bits < 16:
        path = ROOT / args.save
        torch.save({"config": ckpt["config"], "tokenizer": ckpt["tokenizer"], "group": args.group,
                    "packed": {n: {k: v.cpu() if torch.is_tensor(v) else v for k, v in p.items()}
                               for n, p in packed.items()},
                    # params lists the tied embedding once (tok_emb.weight), so it is written once
                    "rest": {n: p.detach().to(torch.bfloat16).cpu() for n, p in params.items() if n not in chosen}},
                   path)
        torch.load(path, weights_only=True)                   # fails if the file needs more than plain data to open
        print(f"saved {path.name}: {path.stat().st_size / 1e6:.1f} MB on disk (computed {size_mb:.1f} MB)")
    if args.sample:
        print(f"{'=' * 25} {bits} bit | {size_mb:.1f} MB {'=' * 25}")
        for p in PROMPTS:
            print(sample(p))
            print("-" * 60)
        continue
    loss = {s: full_loss(d) for s, d in val_sets.items()}
    ref = ref if ref is not None else loss["val"]
    print(f"{bits:2} bit | {bits_per_weight:5.2f} bit/weight | {size_mb:6.1f} MB | val {loss['val']:.4f} "
          f"({loss['val'] - ref:+.4f}) | tr {loss['val_tr'] / chars_per_token['tr']:.4f}/ch "
          f"| en {loss['val_en'] / chars_per_token['en']:.4f}/ch | {time.time() - t0:.0f}s")