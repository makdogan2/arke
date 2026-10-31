# Symmetric (absmax) quantization: first on 8 numbers, then on a real layer of arke0_web.
import torch

def quantize(w, bits):
    """Round every row of w to integers in [-qmax, qmax]; one scale per row."""
    qmax = 2 ** (bits - 1) - 1                              # 8 bit: 127, 4 bit: 7, 2 bit: 1
    scale = w.abs().amax(dim=-1, keepdim=True) / qmax       # the largest |w| of the row maps to qmax
    q = torch.round(w / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

# --- part 1: eight numbers, one scale ---
w = torch.tensor([0.12, -0.40, 0.03, 0.91, -0.07, 0.25, -0.66, 0.002])
for bits in (8, 4, 2):
    q, scale = quantize(w, bits)
    w_hat = dequantize(q, scale)
    print(f"{bits} bit | scale {scale.item():.4f} | q {q.tolist()}")
    print(f"        w_hat {[round(x, 3) for x in w_hat.tolist()]} | mean |error| {(w - w_hat).abs().mean():.4f}")

# --- part 2: a real weight matrix, one scale for the whole matrix vs one per group ---
ckpt = torch.load("arke0_web_last.pt", map_location="cpu", weights_only=False)
W = ckpt["model"]["blocks.0.attn.qkv.weight"].float()     # (2304, 768)
print(f"\nblocks.0.attn.qkv.weight {tuple(W.shape)} | std {W.std():.4f} | max |w| {W.abs().max():.4f}")
for bits in (8, 4, 2):
    row = f"{bits} bit |"
    for name, group in [("whole matrix", W.numel()), ("group 64", 64), ("group 32", 32)]:
        G = W.reshape(-1, group)                               # each row of G is one group
        W_hat = dequantize(*quantize(G, bits)).reshape(W.shape)
        rel = (W - W_hat).norm() / W.norm()                    # relative error: 0 = perfect, 1 = all lost
        row += f" {name}: {rel:.3f} |"
    print(row)