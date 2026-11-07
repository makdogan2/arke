"""How often does a model write the whitespace that textclean.clean() removes from the data?

Every sample starts from <|endoftext|> alone, so the model begins a new document with no prompt
(no prompt choice, and no trailing-space prompt that forces a broken first token).
A document counts as clean if normalize_whitespace(doc).strip() leaves it unchanged: exactly
what data_prep.py wrote between two <|endoftext|>. A run of 8+ whitespace characters is a loop:
the cleaned data never has more than 2 in a row ("\\n\\n").
"""
import argparse, math, re, sys
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import Config, Arke0
from textclean import EOT, normalize_whitespace

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--names", nargs="+", default=["arke0_web_last", "arke0_clean_last"],
                    help="checkpoints <name>.pt to compare")
parser.add_argument("--n", type=int, default=500, help="samples per model")
parser.add_argument("--tokens", type=int, default=256, help="new tokens per sample")
parser.add_argument("--batch", type=int, default=100, help="samples generated together")
parser.add_argument("--seed", type=int, default=1337, help="batch i uses seed + i, the same for every model")
parser.add_argument("--examples", type=int, default=3, help="samples with the longest whitespace runs to show")
args = parser.parse_args()
device = "cuda"

LOOP = 8                                 # whitespace characters in a row that count as a loop
WS_RUN = re.compile(r"\s+")
TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")         # rare in English text: a rough language guess
LEGIT_CUT_ENDS = ("\n\n", "\n", " ")     # what a clean text can end with when cut mid-document

def doc_stats(doc, complete):
    """(characters normalize_whitespace removes, longest whitespace run) for one document."""
    longest = max((len(m) for m in WS_RUN.findall(doc)), default=0)
    if not complete:                      # cut at the token limit: a plain space or line break at the end is fine
        for end in LEGIT_CUT_ENDS:
            if doc.endswith(end) and not doc[:-len(end)].endswith((" ", "\n")):
                doc = doc[:-len(end)]
                break
        clean = normalize_whitespace(doc).lstrip()
    else:
        clean = normalize_whitespace(doc).strip()
    return len(doc) - len(clean), longest

def language(docs):
    """TR or EN by the longest document of the sample."""
    main = max(docs, key=len, default="")
    return "tr" if any(c in TR_LETTERS for c in main) else "en"

@torch.no_grad()
def generate_all(model, eot):
    out = []
    for i in range(math.ceil(args.n / args.batch)):
        b = min(args.batch, args.n - i * args.batch)
        torch.manual_seed(args.seed + i)
        idx = torch.full((b, 1), eot, dtype=torch.long, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):    # as in train.py's eval
            idx = model.generate(idx, args.tokens, temperature=0.8, top_k=50)
        out += idx[:, 1:].tolist()                                       # drop the starting <|endoftext|>
    return out

def pct(k, n):
    p = k / n if n else 0.0
    return f"{100 * p:5.1f}% ±{100 * math.sqrt(p * (1 - p) / n) if n else 0:.1f}"   # binomial std

rows, shown = [], {}
for name in args.names:
    ckpt = torch.load(ROOT / f"{name}.pt", map_location="cpu", weights_only=True)   # only plain data, never code
    model = Arke0(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data" / ckpt["tokenizer"]))
    eot = tok.token_to_id(EOT)
    samples = []
    for ids in generate_all(model, eot):
        text = tok.decode(ids, skip_special_tokens=False)
        docs = text.split(EOT)
        stats = [doc_stats(d, complete=(j < len(docs) - 1)) for j, d in enumerate(docs)]
        samples.append({"text": text, "lang": language(docs), "chars": len(text.replace(EOT, "")),
                        "removed": sum(r for r, _ in stats), "longest": max(l for _, l in stats)})
    for lang in ("all", "tr", "en"):
        s = [x for x in samples if lang == "all" or x["lang"] == lang]
        n = len(s)
        chars = sum(x["chars"] for x in s)
        rows.append((name, lang, n,
                     pct(sum(x["removed"] > 0 for x in s), n),
                     f"{1000 * sum(x['removed'] for x in s) / chars if chars else 0:.2f}",
                     pct(sum(x["longest"] >= LOOP for x in s), n),
                     max((x["longest"] for x in s), default=0)))
    shown[name] = sorted(samples, key=lambda x: -x["longest"])[:args.examples]
    del model, ckpt
    torch.cuda.empty_cache()

print(f"{args.n} samples x {args.tokens} tokens per model, from <|endoftext|>, temperature 0.8, top_k 50, seed {args.seed}")
print(f"{'model':<18} {'lang':<4} {'n':>4} | {'with pattern':>13} | {'removed/1k ch':>13} | {'loop (8+)':>13} | longest")
for name, lang, n, pat, rem, loop, longest in rows:
    print(f"{name:<18} {lang:<4} {n:4} | {pat:>13} | {rem:>13} | {loop:>13} | {longest:7}")

for name, s in shown.items():
    print(f"\n=== {name}: the {len(s)} samples with the longest whitespace runs")
    for x in s:
        text = x["text"]
        runs = [m for m in WS_RUN.finditer(text)]
        m = max(runs, key=lambda m: len(m.group()), default=None)
        if m is None:
            continue
        a, b = max(0, m.start() - 40), min(len(text), m.end() + 40)
        print(f"[{x['lang']}, run {x['longest']}, removed {x['removed']}] {text[a:b]!r}")
