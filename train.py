import argparse, json, math, time
import torch
from pathlib import Path
from dataclasses import asdict
from tokenizers import Tokenizer
from model import Config, Arke0

ROOT = Path(__file__).parent
TOKENIZER_FILE = "bpe_8192.json"

# model size from the command line; defaults come from Config, so plain `train.py` is unchanged
parser = argparse.ArgumentParser()
parser.add_argument("--n_layer", type=int, default=Config.n_layer)
parser.add_argument("--n_embd",  type=int, default=Config.n_embd)
parser.add_argument("--n_head",  type=int, default=Config.n_head)
parser.add_argument("--name", default="arke0_bpe", help="checkpoint <name>.pt and log runs/<name>.json")
args = parser.parse_args()

d = torch.load(ROOT / "data" / "bpe.pt", weights_only=False)
train_data, val_data = d["train"], d["val"]
tok = Tokenizer.from_file(str(ROOT / "data" / TOKENIZER_FILE))

cfg = Config(vocab_size=d["vocab_size"], n_layer=args.n_layer,
             n_embd=args.n_embd, n_head=args.n_head)

batch_size = 64
max_iters  = 8000
warmup     = 200
eval_every = 500
max_lr     = 6e-4
min_lr     = 6e-5
device     = "cuda"

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

def get_lr(it):
    if it < warmup:                                  # linear warmup
        return max_lr * (it + 1) / warmup
    ratio = (it - warmup) / (max_iters - warmup)     # cosine decay
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+cfg.block_size].long() for i in ix])
    y = torch.stack([data[i+1:i+cfg.block_size+1].long() for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(50)
        for i in range(50):
            X, Y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

model = Arke0(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())   # tied head is counted once
# scaling laws use parameters without the embedding lookup tables
n_params_non_emb = n_params - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()
print(f"config: {asdict(cfg)}")
print(f"parameters: {n_params/1e6:.2f}M ({n_params_non_emb/1e6:.2f}M without embeddings)")
print(f"tokens seen at end: {max_iters*batch_size*cfg.block_size/1e6:.0f}M "
      f"({max_iters*batch_size*cfg.block_size/len(train_data):.1f} epochs)")

opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

history = []                                         # one entry per eval, saved to runs/<name>.json
t0 = time.time()
for it in range(max_iters + 1):
    lr = get_lr(it)
    for group in opt.param_groups:
        group["lr"] = lr

    if it % eval_every == 0:
        l = estimate_loss()
        history.append({"step": it, "train": l["train"], "val": l["val"],
                        "lr": lr, "time": time.time() - t0})
        print(f"step {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} "
              f"| lr {lr:.1e} | {time.time()-t0:.0f}s")

    X, Y = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
train_time = time.time() - t0

ckpt = ROOT / f"{args.name}.pt"
tmp = ckpt.with_suffix(".tmp")
torch.save({"model": model.state_dict(),
            "config": asdict(cfg),            # the instance's real values
            "tokenizer": TOKENIZER_FILE,      # which tokenizer this model speaks
            "step": max_iters,
            "val_loss": l["val"]},
           tmp)
tmp.replace(ckpt)
print("saved:", ckpt.name)

log = ROOT / "runs" / f"{args.name}.json"
log.parent.mkdir(exist_ok=True)
log.write_text(json.dumps({"name": args.name,
                           "config": asdict(cfg),
                           "params": n_params,
                           "params_non_emb": n_params_non_emb,
                           "batch_size": batch_size,
                           "max_iters": max_iters,
                           "max_lr": max_lr,
                           "train_time": train_time,
                           "history": history}, indent=2))
print("log:", log.relative_to(ROOT))

start = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(start, 300, temperature=0.8, top_k=50)[0].tolist()
print("\n--- Arke speaks ---")
print(tok.decode(out, skip_special_tokens=False).replace("<|endoftext|>", "\n\n<|endoftext|>\n\n"))