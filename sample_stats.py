"""How often does a model write the whitespace that textclean.clean() removes, and how much does it repeat itself?

By default every sample starts from <|endoftext|> alone, so the model begins a new document with no prompt
(no prompt choice, and no trailing-space prompt that forces a broken first token); --prompts starts from text.
A document counts as clean if normalize_whitespace(doc).strip() leaves it unchanged: exactly
what data_prep.py wrote between two <|endoftext|>. A run of 8+ whitespace characters is a loop:
the cleaned data never has more than 2 in a row ("\\n\\n").

Repetition is measured on words: seq-rep-4 is the share of a sample's word 4-grams that already appeared in
the same document, and a word loop is one word 3+ times in a row. Real text repeats too, so evenly spaced
validation windows of the same length get the same two numbers as a reference.
"""
import argparse, itertools, json, math, re, sys
from pathlib import Path
import numpy as np
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
parser.add_argument("--prompts", nargs="+", default=None,
                    help="start from these texts, with no <|endoftext|> before them (as sample.py does); "
                         "the samples are split evenly between them")
parser.add_argument("--data", default="data/v2", help="folder with val.bin and meta.json for the real-text reference")
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
device = "cuda"

LOOP = 8                                 # whitespace characters in a row that count as a loop
WS_RUN = re.compile(r"\s+")
TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")         # rare in English text: a rough language guess
TR_SHARE = 0.01                          # TR if at least this share of the letters are TR_LETTERS
LETTER = re.compile(r"[^\W\d_]")
LEGIT_CUT_ENDS = ("\n\n", "\n", " ")     # what a clean text can end with when cut mid-document
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")   # letters, with an apostrophe inside: Rusya’dan, don't
NGRAM = 4                                # seq-rep-n of Welleck et al. 2019
WORD_LOOP = 3                            # the same word this many times in a row counts as a loop

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

def repetition(docs):
    """(repeated 4-grams, all 4-grams, longest run of one word) over the documents of a sample."""
    repeated, total, run = 0, 0, 0
    for doc in docs:
        words = [w.lower() for w in WORD.findall(doc)]
        grams = [tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)]
        repeated += len(grams) - len(set(grams))
        total += len(grams)
        run = max(run, max((len(list(g)) for _, g in itertools.groupby(words)), default=0))
    return repeated, total, run

def language(docs):
    """TR or EN by the longest document of the sample. A share, not a single letter: English samples
    often carry a few Turkish words (median share in Turkish text 10.5%, in English 0)."""
    main = max(docs, key=len, default="")
    letters = LETTER.findall(main)
    tr = sum(c in TR_LETTERS for c in letters)
    return "tr" if letters and tr / len(letters) >= TR_SHARE else "en"

@torch.no_grad()
def generate_all(model, tok, eot):
    starts = [tok.encode(p).ids for p in args.prompts] if args.prompts else [[eot]]
    drop = 0 if args.prompts else 1                                      # the starting <|endoftext|> is not text
    per_start = args.n // len(starts)
    out, seed = [], args.seed
    for start in starts:
        for first in range(0, per_start, args.batch):
            b = min(args.batch, per_start - first)
            torch.manual_seed(seed)
            seed += 1
            idx = torch.tensor([start] * b, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):    # as in train.py's eval
                idx = model.generate(idx, args.tokens, temperature=args.temperature,
                                     top_k=args.top_k or None, top_p=args.top_p,
                                     repetition_penalty=args.repetition_penalty)
            out += idx[:, drop:].tolist()
    return out

def pct(k, n):
    p = k / n if n else 0.0
    return f"{100 * p:5.1f}% ±{100 * math.sqrt(p * (1 - p) / n) if n else 0:.1f}"   # binomial std

def mean_pct(values):
    n = len(values)
    if n < 2:
        return "-"
    m = sum(values) / n
    return f"{100 * m:5.1f}% ±{100 * math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1) / n):.1f}"   # std of the mean

def score(text):
    """Everything measured on one sample (or one window of real text)."""
    docs = text.split(EOT)
    stats = [doc_stats(d, complete=(j < len(docs) - 1)) for j, d in enumerate(docs)]
    repeated, total, word_run = repetition(docs)
    return {"text": text, "lang": language(docs), "chars": len(text.replace(EOT, "")),
            "removed": sum(r for r, _ in stats), "longest": max(l for _, l in stats),
            "rep": repeated / total if total else None, "word_run": word_run}

def repetition_columns(s):
    return mean_pct([x["rep"] for x in s if x["rep"] is not None]), pct(sum(x["word_run"] >= WORD_LOOP for x in s), len(s))

# real text first: the same windows for every run, so the reference never moves
data_dir = ROOT / args.data
meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
val_tok = Tokenizer.from_file(str(data_dir / meta["tokenizer"]))
gap = (len(val) - args.tokens) // args.n
real = [score(val_tok.decode(val[i * gap:i * gap + args.tokens].tolist(), skip_special_tokens=False)) for i in range(args.n)]
rows = []
for lang in ("all", "tr", "en"):
    s = [x for x in real if lang == "all" or x["lang"] == lang]
    rows.append(("real val text", lang, len(s), "-", "-", "-", "-", *repetition_columns(s)))

shown = {}
for name in args.names:
    ckpt = torch.load(ROOT / f"{name}.pt", map_location="cpu", weights_only=True)   # only plain data, never code
    model = Arke0(Config(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data" / ckpt["tokenizer"]))
    eot = tok.token_to_id(EOT)
    samples = [score(tok.decode(ids, skip_special_tokens=False)) for ids in generate_all(model, tok, eot)]
    for lang in ("all", "tr", "en"):
        s = [x for x in samples if lang == "all" or x["lang"] == lang]
        n = len(s)
        chars = sum(x["chars"] for x in s)
        rows.append((name, lang, n,
                     pct(sum(x["removed"] > 0 for x in s), n),
                     f"{1000 * sum(x['removed'] for x in s) / chars if chars else 0:.2f}",
                     pct(sum(x["longest"] >= LOOP for x in s), n),
                     max((x["longest"] for x in s), default=0),
                     *repetition_columns(s)))
    shown[name] = sorted(samples, key=lambda x: -x["longest"])[:args.examples]
    del model, ckpt
    torch.cuda.empty_cache()

source = f"from {args.prompts}" if args.prompts else "from <|endoftext|>"
print(f"{args.n} samples x {args.tokens} tokens per model, {source}, temperature {args.temperature}, "
      f"top_k {args.top_k}, top_p {args.top_p}, repetition_penalty {args.repetition_penalty}, seed {args.seed}")
print(f"{'model':<18} {'lang':<4} {'n':>4} | {'with pattern':>13} | {'removed/1k ch':>13} | {'loop (8+)':>13} | {'longest':>7} "
      f"| {'seq-rep-4':>13} | {'word loop':>13}")
for name, lang, n, pat, rem, loop, longest, rep, word_loop in rows:
    print(f"{name:<18} {lang:<4} {n:4} | {pat:>13} | {rem:>13} | {loop:>13} | {longest:>7} | {rep:>13} | {word_loop:>13}")

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
