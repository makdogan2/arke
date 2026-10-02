import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import Config, UMAY0

ROOT = Path(__file__).parent
CKPT = ROOT / "umay0_bpe.pt"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

ckpt = torch.load(CKPT, map_location=device, weights_only=False)

# Rebuild the exact architecture the checkpoint was trained with
cfg = Config(**ckpt["config"])
tok = Tokenizer.from_file(str(ROOT / "data" / ckpt.get("tokenizer", "bpe_8192.json")))

model = UMAY0(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"device: {device} | config: {ckpt['config']}")
if "val_loss" in ckpt:
    print(f"trained {ckpt['step']} steps, val loss {ckpt['val_loss']:.4f}")

def generate(prompt, n=200, temperature=0.8, top_k=50):
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature, top_k)[0].tolist()
    return tok.decode(out)

for p in ["Türkiye, ", "Ankara şehri ", "The history of ", "Bilim insanları "]:
    print("=" * 60)
    print(generate(p))