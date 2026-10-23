import argparse, torch, time, math
from pathlib import Path
from dataclasses import asdict
from tokenizers import Tokenizer
from model import Config, Arke0

ROOT = Path(__file__).parent
TOKENIZER_FILE = "bpe_8192.json"

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="arke0_bpe",
                    help="checkpoint name: <out>.pt is the best model, <out>_last.pt the latest state")
parser.add_argument("--resume", action="store_true", help="continue from <out>_last.pt")
parser.add_argument("--max-iters", type=int, default=8000)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--stop-at", type=int, default=None,
                    help="quit at this step without finishing (simulates a crash, for testing --resume)")
args = parser.parse_args()

d = torch.load(ROOT / "data" / "bpe.pt", weights_only=False)
train_data, val_data = d["train"], d["val"]
tok = Tokenizer.from_file(str(ROOT / "data" / TOKENIZER_FILE))

batch_size = 64
max_iters  = args.max_iters
warmup     = 200
eval_every = args.eval_every
save_every = args.save_every
max_lr     = 6e-4
min_lr     = 6e-5
device     = "cuda"

ckpt_best = ROOT / f"{args.out}.pt"
ckpt_last = ROOT / f"{args.out}_last.pt"

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

def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

def save(path, step):
    """Write everything needed to rebuild the model AND to continue training exactly.
    `step` = number of optimizer updates already done."""
    state = {
        # TODO(Mehmet): fill in the dict. sample.py reads "model", "config", "tokenizer", "step".
        #   model weights, optimizer state (AdamW's moving averages!), the full config as a dict,
        #   step, best_val, the tokenizer file name,
        #   "train": the hyperparameters that shape the run (batch_size, max_iters, warmup, max_lr, min_lr),
        #   "rng": the CPU and CUDA random generator states (torch.get_rng_state / torch.cuda.get_rng_state)
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)          # atomic: a crash while writing never corrupts the previous file

if args.resume:
    # load to CPU: RNG states must stay CPU tensors; weights and optimizer state
    # are copied to the GPU by load_state_dict
    ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False)
    cfg = ...                          # TODO: rebuild the architecture from the checkpoint, not from defaults
    max_iters = ...                    # TODO: from the checkpoint, so the cosine schedule stays the same
    model = Arke0(cfg).to(device)
    model.load_state_dict(...)         # TODO
    opt = make_optimizer(model)
    opt.load_state_dict(...)           # TODO
    start_step = ...                   # TODO
    best_val = ...                     # TODO
    torch.set_rng_state(...)           # TODO: same CPU generator -> get_batch() draws the same batches
    torch.cuda.set_rng_state(...)      # TODO: same CUDA generator -> same dropout masks
    print(f"resumed {ckpt_last.name} at step {start_step} | best val {best_val:.4f}")
else:
    cfg = Config(vocab_size=d["vocab_size"])
    model = Arke0(cfg).to(device)
    opt = make_optimizer(model)
    start_step, best_val = 0, float("inf")

n_params = sum(p.numel() for p in model.parameters())
print(f"config: {asdict(cfg)}")
print(f"parameters: {n_params/1e6:.2f}M")
print(f"tokens seen at end: {max_iters*batch_size*cfg.block_size/1e6:.0f}M "
      f"({max_iters*batch_size*cfg.block_size/len(train_data):.1f} epochs)")

t0 = time.time()
for it in range(start_step, max_iters + 1):
    # TODO(Mehmet): every save_every steps, save the latest state to ckpt_last.
    #   It happens HERE, before eval and the update: the file then means "it updates done",
    #   and a resume starts this loop at exactly the same `it`.
    #   Skip it == start_step: that state is already on disk (or is the untrained model).
    if ...:
        save(ckpt_last, it)

    if args.stop_at is not None and it == args.stop_at:
        print(f"--stop-at {it}: quitting without finishing")
        raise SystemExit

    lr = get_lr(it)
    for group in opt.param_groups:
        group["lr"] = lr

    if it % eval_every == 0:
        l = estimate_loss()
        print(f"step {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} "
              f"| lr {lr:.1e} | {time.time()-t0:.0f}s")
        # TODO(Mehmet): if this is the lowest val loss so far, update best_val and save ckpt_best
        ...

    if it == max_iters:        # the last step only evaluates
        break

    X, Y = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

save(ckpt_last, max_iters)
print(f"saved: {ckpt_last.name} | best: {ckpt_best.name} (val {best_val:.4f})")

start = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(start, 300, temperature=0.8, top_k=50)[0].tolist()
print("\n--- Arke speaks ---")
print(tok.decode(out, skip_special_tokens=False).replace("<|endoftext|>", "\n\n<|endoftext|>\n\n"))
