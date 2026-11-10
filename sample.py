import argparse, sys
import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import Config, Arke0

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Turkish text also when the output goes to a file
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="arke0_bpe", help="checkpoint <name>.pt written by train.py")
parser.add_argument("--seed", type=int, default=1337, help="each prompt starts from this seed: the same command gives the same draws")
args = parser.parse_args()
CKPT = ROOT / f"{args.name}.pt"
EOT = "<|endoftext|>"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

ckpt = torch.load(CKPT, map_location=device, weights_only=True)

# Rebuild the exact architecture the checkpoint was trained with
cfg = Config(**ckpt["config"])
tok = Tokenizer.from_file(str(ROOT / "data" / ckpt.get("tokenizer", "bpe_8192.json")))

model = Arke0(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"device: {device} | config: {ckpt['config']}")
if ckpt.get("history"):                            # this checkpoint's own val, not the best one of the run
    val, val_step = ckpt["history"][-1]["val"], ckpt["history"][-1]["step"]
else:                                              # older checkpoints have no history; the oldest stored "val_loss"
    val, val_step = ckpt.get("best_val", ckpt.get("val_loss")), ckpt["step"]
if val is not None:
    print(f"trained {ckpt['step']} steps, val loss {val:.4f} (measured at step {val_step})")

def generate(prompt, n=200, temperature=0.8, top_k=50):
    torch.manual_seed(args.seed)
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate(idx, n, temperature, top_k)[0].tolist()
    # keep <|endoftext|> visible: the model ends a document there and starts a new one
    return tok.decode(out, skip_special_tokens=False).replace(EOT, f"\n\n{EOT}\n\n")

# no space at the end: a lone space token would force the model to begin with a broken word
for p in ["Türkiye,", "Bilim insanları", "The history of", "Scientists have"]:
    print("=" * 60)
    print(generate(p))