import argparse, importlib.util, json, math, os, signal, sys, time
from datetime import datetime, timedelta
import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict
from tokenizers import Tokenizer
from model import Config, Arke0

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Turkish text also when the output goes to a file

# model size from the command line; defaults come from Config, so plain `train.py` is unchanged
parser = argparse.ArgumentParser()
parser.add_argument("--n_layer", type=int, default=Config.n_layer)
parser.add_argument("--n_embd",  type=int, default=Config.n_embd)
parser.add_argument("--n_head",  type=int, default=Config.n_head)
parser.add_argument("--name", default="arke0_bpe",
                    help="<name>.pt is the best model, <name>_last.pt the latest state, runs/<name>.json the log")
parser.add_argument("--data", default="data",
                    help="folder with train.bin, val.bin and meta.json from data_prep.py")
parser.add_argument("--resume", action="store_true",
                    help="continue from <name>_last.pt (model size, data and schedule come from the checkpoint)")
parser.add_argument("--max-iters", type=int, default=8000)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--save-every", type=int, default=500)
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the model (needs Triton; on Windows: uv pip install triton-windows)")
parser.add_argument("--stop-at", type=int, default=None,
                    help="quit at this step without finishing (simulates a crash, for testing --resume)")
parser.add_argument("--hours", type=float, default=None,
                    help="stop this session after this many hours: save, exit, continue later with --resume")
parser.add_argument("--until", default=None,
                    help="stop this session at this clock time, e.g. 08:00 (tomorrow if it has passed today)")
args = parser.parse_args()

batch_size = 64
max_iters  = args.max_iters
warmup     = 200
eval_every = args.eval_every
save_every = args.save_every
max_lr     = 6e-4
min_lr     = 6e-5
device     = "cuda"

ckpt_best = ROOT / f"{args.name}.pt"
ckpt_last = ROOT / f"{args.name}_last.pt"
# load to CPU: RNG states must stay CPU tensors; weights and optimizer state
# are copied to the GPU by load_state_dict
ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=False) if args.resume else None

def rel(path, base):
    """Path as stored in checkpoints: relative, so the repo can move."""
    return Path(os.path.relpath(path, base)).as_posix()

# token ids as uint16 on disk; memmap reads only the windows we sample, the file never has to fit in RAM
data_dir = ROOT / (ckpt["data"] if ckpt else args.data)
meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap(data_dir / "val.bin",   dtype=np.uint16, mode="r")
tok_file = data_dir / meta["tokenizer"]
tok = Tokenizer.from_file(str(tok_file))

if args.compile and importlib.util.find_spec("triton") is None:
    print("torch.compile needs Triton (on Windows: uv pip install triton-windows), running without it")
    args.compile = False

torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")

def get_lr(it):
    if it < warmup:                                  # linear warmup
        return max_lr * (it + 1) / warmup
    ratio = (it - warmup) / (max_iters - warmup)     # cosine decay
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size - 1, (batch_size,)).tolist()   # torch RNG: saved in checkpoints
    # one read of block_size+1 tokens per window: x is the window, y the same window shifted by one
    buf = torch.from_numpy(np.stack([data[i:i + cfg.block_size + 1] for i in ix]).astype(np.int64))
    buf = buf.pin_memory().to(device, non_blocking=True)   # pinned memory: the copy runs while the CPU moves on
    return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(50)
        for i in range(50):
            X, Y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = fwd(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def make_optimizer(model):
    # fused: one GPU kernel updates all parameters instead of several small kernels per tensor
    return torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)

def replace_retry(tmp, path, tries=10):
    """tmp.replace(path), but on Windows a virus scanner or the search indexer can hold the
    file open for a moment and the rename fails with PermissionError: wait and try again."""
    for attempt in range(1, tries + 1):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == tries:
                raise
            time.sleep(0.1 * attempt)

def save(path, step):
    """Write everything needed to rebuild the model AND to continue training exactly.
    `step` = number of optimizer updates already done."""
    state = {
        "model": model.state_dict(),          # the plain model, not the compiled wrapper
        "optimizer": opt.state_dict(),        # AdamW's moving averages: without them a resume restarts "cold"
        "config": asdict(cfg),                # the instance's real values
        "tokenizer": rel(tok_file, ROOT / "data"),   # which tokenizer this model speaks (sample.py looks in data/)
        "data": rel(data_dir, ROOT),
        "step": step,
        "best_val": best_val,
        "train": dict(batch_size=batch_size, max_iters=max_iters, warmup=warmup,
                      max_lr=max_lr, min_lr=min_lr, eval_every=eval_every, save_every=save_every),
        "history": history,
        "elapsed": time.time() - t0,
        "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    replace_retry(tmp, path)   # atomic: a crash while writing never corrupts the previous file

if ckpt:
    cfg = Config(**ckpt["config"])     # the checkpoint's architecture, not the defaults / command line
    tr = ckpt["train"]                 # same schedule as before the interruption
    batch_size, max_iters, warmup, max_lr, min_lr = (
        tr[k] for k in ("batch_size", "max_iters", "warmup", "max_lr", "min_lr"))
    # evals draw batches from the same RNG, so a different eval schedule would change the training batches
    eval_every, save_every = tr.get("eval_every", eval_every), tr.get("save_every", save_every)
    model = Arke0(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    opt = make_optimizer(model)
    opt.load_state_dict(ckpt["optimizer"])
    start_step, best_val, history = ckpt["step"], ckpt["best_val"], ckpt["history"]
    t0 = time.time() - ckpt["elapsed"]
    torch.set_rng_state(ckpt["rng"]["cpu"])         # same CPU generator -> get_batch() draws the same batches
    torch.cuda.set_rng_state(ckpt["rng"]["cuda"])   # same CUDA generator -> same dropout masks
    print(f"resumed {ckpt_last.name} at step {start_step} | best val {best_val:.4f}")
else:
    cfg = Config(vocab_size=meta["vocab_size"], n_layer=args.n_layer,
                 n_embd=args.n_embd, n_head=args.n_head)
    model = Arke0(cfg).to(device)
    opt = make_optimizer(model)
    start_step, best_val = 0, float("inf")
    history = []                                     # one entry per eval, saved to runs/<name>.json
    t0 = time.time()

# compile fuses small GPU ops (layernorm, gelu, adds) into bigger kernels; the first steps are slow
# because it compiles then. Weights stay shared with `model`, which we save and use for generate().
fwd = torch.compile(model) if args.compile else model

n_params = sum(p.numel() for p in model.parameters())   # tied head is counted once
# scaling laws use parameters without the embedding lookup tables
n_params_non_emb = n_params - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()
print(f"config: {asdict(cfg)} | data: {rel(data_dir, ROOT)} | compile: {args.compile}")
print(f"parameters: {n_params/1e6:.2f}M ({n_params_non_emb/1e6:.2f}M without embeddings)")
print(f"tokens seen at end: {max_iters*batch_size*cfg.block_size/1e6:.0f}M "
      f"({max_iters*batch_size*cfg.block_size/len(train_data):.1f} epochs)")

def write_run_log():
    """runs/<name>.json: settings and loss history so far, rewritten at every stop and at the end."""
    log = ROOT / "runs" / f"{args.name}.json"
    log.parent.mkdir(exist_ok=True)
    log.write_text(json.dumps({"name": args.name,
                               "config": asdict(cfg),
                               "data": rel(data_dir, ROOT),
                               "params": n_params,
                               "params_non_emb": n_params_non_emb,
                               "batch_size": batch_size,
                               "max_iters": max_iters,
                               "max_lr": max_lr,
                               "train_time": time.time() - t0,
                               "best_val": best_val,
                               "history": history}, indent=2))
    return log

# When this session ends: --hours / --until set a deadline, Ctrl+C asks for a stop.
# Either way the loop stops at the top of a step, saves <name>_last.pt and exits; --resume continues.
deadline = None
if args.hours is not None:
    deadline = time.time() + args.hours * 3600
if args.until is not None:
    h, m = map(int, args.until.split(":"))
    end = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= datetime.now():
        end += timedelta(days=1)
    deadline = min(deadline, end.timestamp()) if deadline else end.timestamp()
if deadline:
    print(f"this session stops at {datetime.fromtimestamp(deadline):%d.%m %H:%M}")

stop_requested = False
def request_stop(signum, frame):
    """First Ctrl+C: finish the current step, save, exit. A second Ctrl+C quits at once."""
    global stop_requested
    stop_requested = True
    signal.signal(signal.SIGINT, signal.default_int_handler)
    print("\nCtrl+C: saving and stopping at the next step (Ctrl+C again: quit without saving)")
signal.signal(signal.SIGINT, request_stop)

mark_t, mark_it, tok_s = time.time(), start_step, 0.0      # training speed, without eval and saving
for it in range(start_step, max_iters + 1):
    if it % eval_every == 0 and it > mark_it:
        torch.cuda.synchronize()                    # GPU runs ahead of Python: wait before reading the clock
        tok_s = (it - mark_it) * batch_size * cfg.block_size / (time.time() - mark_t)

    # Save here, before eval and the update: the file then means "it updates done",
    # and a resume starts this loop at exactly the same `it`. At it == start_step
    # that state is already on disk (or is the untrained model).
    if it % save_every == 0 and it > start_step:
        save(ckpt_last, it)

    if args.stop_at is not None and it == args.stop_at:
        print(f"--stop-at {it}: quitting without finishing")
        raise SystemExit

    if stop_requested or (deadline and time.time() >= deadline):
        if it > start_step and it % save_every != 0:   # a multiple of save_every was saved just above
            save(ckpt_last, it)
        write_run_log()
        print(f"stopped at step {it}/{max_iters}, continue with: train.py --name {args.name} --resume")
        raise SystemExit

    lr = get_lr(it)
    for group in opt.param_groups:
        group["lr"] = lr

    if it % eval_every == 0:
        l = estimate_loss()
        history.append({"step": it, "train": l["train"], "val": l["val"],
                        "lr": lr, "time": time.time() - t0, "tok_s": tok_s})
        print(f"step {it:5d} | train {l['train']:.4f} | val {l['val']:.4f} "
              f"| lr {lr:.1e} | {tok_s/1e3:5.0f}k tok/s | {time.time()-t0:.0f}s")
        if l["val"] < best_val:
            best_val = l["val"]
            save(ckpt_best, it)
        mark_t, mark_it = time.time(), it

    if it == max_iters:        # the last step only evaluates
        break

    X, Y = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = fwd(X, Y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

save(ckpt_last, max_iters)
print(f"saved: {ckpt_last.name} | best: {ckpt_best.name} (val {best_val:.4f})")
print("log:", write_run_log())

start = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(start, 300, temperature=0.8, top_k=50)[0].tolist()
print("\n--- Arke speaks ---")
print(tok.decode(out, skip_special_tokens=False).replace("<|endoftext|>", "\n\n<|endoftext|>\n\n"))