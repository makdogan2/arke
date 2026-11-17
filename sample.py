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
# no space at the end of a prompt: a lone space token would force the model to begin with a broken word
parser.add_argument("--prompts", nargs="+", default=["Türkiye,", "Bilim insanları", "The history of", "Scientists have"],
                    help="texts to continue, each without a space at the end")
parser.add_argument("--tokens", type=int, default=200, help="new tokens per prompt at most")
parser.add_argument("--temperature", type=float, default=0.8,
                    help="divides the logits before sampling: below 1 favors likely tokens, above 1 flattens")
parser.add_argument("--top_k", type=int, default=50, help="sample only among the k most likely tokens; 0 turns it off")
parser.add_argument("--top_p", type=float, default=None,
                    help="nucleus sampling: keep the most likely tokens until their probabilities reach this sum")
parser.add_argument("--repetition_penalty", type=float, default=None,
                    help="tokens already in the context get less likely (1.2 is common); off by default")
args = parser.parse_args()
if args.temperature <= 0:
    parser.error("--temperature must be above 0: generate divides the logits by it")
if args.top_p is not None and not 0 < args.top_p <= 1:
    parser.error("--top_p must be above 0 and at most 1")
if args.repetition_penalty is not None and args.repetition_penalty <= 0:
    parser.error("--repetition_penalty must be above 0: generate divides the logits by it")
if args.top_k < 0:
    parser.error("--top_k must be 0 (off) or a positive number of tokens")
if args.tokens < 1:
    parser.error("--tokens must be at least 1")
CKPT = ROOT / f"{args.name}.pt"
EOT = "<|endoftext|>"

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

ckpt = torch.load(CKPT, map_location=device, weights_only=True)

# Rebuild the exact architecture the checkpoint was trained with
cfg = Config(**ckpt["config"])
if args.top_k > cfg.vocab_size:
    parser.error(f"--top_k cannot be larger than the vocabulary ({cfg.vocab_size} tokens); 0 turns it off")
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

def generate(prompt):
    torch.manual_seed(args.seed)
    ids = tok.encode(prompt).ids
    if not ids:
        sys.exit("a prompt must have at least one token: the model continues a text, it does not start from nothing")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    # stop at the end of the document instead of starting a new, unrelated one
    out = model.generate(idx, args.tokens, args.temperature, args.top_k or None, stop_token=tok.token_to_id(EOT),
                         top_p=args.top_p, repetition_penalty=args.repetition_penalty)[0].tolist()
    # keep <|endoftext|> visible: it shows the model ended the document itself, not the token limit
    return tok.decode(out, skip_special_tokens=False).replace(EOT, f"\n{EOT}")

for p in args.prompts:
    print("=" * 60)
    print(generate(p))