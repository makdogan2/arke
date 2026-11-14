"""Blind pairwise comparison of two sampling settings, judged by a person.

Each pair continues the same prompt, the beginning of a real validation document (half TR, half EN),
once with each setting. Which setting is A and which is B is drawn at random per pair and written to a
separate key file, so the reader cannot tell them apart.

    python blind_pairs.py --name arke0_clean_last          # writes runs/blind/pairs.txt and key.json
    (write A, B or = after "choice:" for every pair in pairs.txt)
    python blind_pairs.py --score                           # counts the wins of each setting
    python blind_pairs.py --score --choices judge.txt --agree pairs.txt   # another judge, and how often they agree

A setting is three or four numbers: temperature, top_k (0 = off), top_p (0 = off) and, optionally,
repetition_penalty (0 or left out = off), e.g. --second 0.8 50 0 1.2.
"""
import argparse, json, math, random, re, sys
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer
from model import Config, Arke0
from textclean import EOT

ROOT = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--name", default="arke0_clean_last", help="checkpoint <name>.pt")
parser.add_argument("--first", nargs="+", type=float, default=[0.8, 50, 0], metavar="X",
                    help="first setting: temperature, top_k (0 = off), top_p (0 = off), "
                         "optionally repetition_penalty (0 = off)")
parser.add_argument("--second", nargs="+", type=float, default=[1.0, 0, 0.9], metavar="X",
                    help="second setting, in the same form")
parser.add_argument("--n", type=int, default=50, help="pairs, split evenly between TR and EN prompts")
parser.add_argument("--tokens", type=int, default=100, help="new tokens per continuation at most")
parser.add_argument("--prompt_tokens", type=int, default=16, help="prompt length, shortened to end at a word")
parser.add_argument("--seed", type=int, default=1337, help="pair i samples both settings from seed + i")
parser.add_argument("--data", default="data/v2", help="folder with val_tr.bin, val_en.bin and meta.json")
parser.add_argument("--out", default="runs/blind", help="folder for pairs.txt and key.json")
parser.add_argument("--score", action="store_true", help="count the choices instead of writing the pairs")
parser.add_argument("--choices", default="pairs.txt", help="--score: the file in --out with the choices to count")
parser.add_argument("--agree", default=None,
                    help="--score: another choices file in --out; prints how often the two judges agree")
args = parser.parse_args()
for s in (args.first, args.second):
    if len(s) not in (3, 4):
        parser.error("a setting is 3 or 4 numbers: temperature, top_k, top_p and optionally repetition_penalty")
args.first, args.second = (s + [0.0] * (4 - len(s)) for s in (args.first, args.second))   # no penalty: 0
out = ROOT / args.out
PAIR = re.compile(r"^=== pair (\d+)", re.M)
CHOICE = re.compile(r"^choice:[ \t]*(\S*)", re.M)

def label(s):
    t, k, p, r = list(s) + [0] * (4 - len(s))          # key.json files from before the penalty have 3 numbers
    return (f"T {t:g}" + (f", top-k {k:g}" if k else "") + (f", top-p {p:g}" if p else "")
            + (f", penalty {r:g}" if r else ""))

def sign_test(a, b):
    """Two-sided exact binomial p-value: how likely a split at least this uneven is if both are equally good."""
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def read_choices(name, n):
    """(pair numbers, choices) from a filled copy of pairs.txt; stops on a missing or unknown choice."""
    text = (out / name).read_text(encoding="utf-8")
    pairs = [int(m.group(1)) for m in PAIR.finditer(text)]
    choices = [m.group(1).upper() for m in CHOICE.finditer(text)]
    if len(pairs) != n or len(choices) != len(pairs):
        sys.exit(f"{len(pairs)} pairs and {len(choices)} choice lines in {name}, the key has {n}")
    if any(c not in ("A", "B", "=") for c in choices):
        sys.exit(f"every choice in {name} must be A, B or =; still open: "
                 + ", ".join(str(p) for p, c in zip(pairs, choices) if c not in ("A", "B", "=")))
    return pairs, choices

if args.score:
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    pairs, choices = read_choices(args.choices, len(key["pairs"]))
    print(f"{key['name']}: first = {label(key['first'])}, second = {label(key['second'])} ({args.choices})")
    for lang in ("all", "tr", "en"):
        wins = {"first": 0, "second": 0, "tie": 0}
        for p, c in zip(pairs, choices):
            info = key["pairs"][p - 1]
            if lang != "all" and info["lang"] != lang:
                continue
            wins["tie" if c == "=" else info[c]] += 1
        print(f"{lang:<4} first {wins['first']:3} | second {wins['second']:3} | tie {wins['tie']:3} "
              f"| sign test p = {sign_test(wins['first'], wins['second']):.3f}")
    if args.agree:
        other_pairs, other = read_choices(args.agree, len(key["pairs"]))
        if other_pairs != pairs:
            sys.exit(f"{args.choices} and {args.agree} list different pairs")
        same = sum(a == b for a, b in zip(choices, other))
        both = [(a, b) for a, b in zip(choices, other) if a != "=" and b != "="]   # both picked a side
        print(f"agreement with {args.agree}: same choice in {same}/{len(pairs)} pairs; "
              f"where both picked a side, the same side in {sum(a == b for a, b in both)}/{len(both)}")
    sys.exit()

if (out / "pairs.txt").exists():
    sys.exit(f"{out / 'pairs.txt'} already exists: it may hold your choices. Move it or pass another --out.")
for s in (args.first, args.second):
    if s[0] <= 0 or s[1] < 0 or not 0 <= s[2] <= 1 or s[3] < 0:
        parser.error("a setting needs temperature > 0, top_k >= 0, top_p between 0 and 1 and repetition_penalty >= 0")
device = "cuda"
ckpt = torch.load(ROOT / f"{args.name}.pt", map_location="cpu", weights_only=True)   # only plain data, never code
model = Arke0(Config(**ckpt["config"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer.from_file(str(ROOT / "data" / ckpt["tokenizer"]))
eot = tok.token_to_id(EOT)
data_dir = ROOT / args.data
meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
if (ROOT / "data" / ckpt["tokenizer"]).read_bytes() != (data_dir / meta["tokenizer"]).read_bytes():
    sys.exit(f"{args.data} was tokenized with {meta['tokenizer']}, not the model's {ckpt['tokenizer']}")

def prompts(lang, n):
    """The first prompt_tokens tokens of n evenly spaced documents, cut back to the end of a word."""
    val = np.memmap(data_dir / f"val_{lang}.bin", dtype=np.uint16, mode="r")
    ends = np.flatnonzero(val == eot)
    starts = np.concatenate(([0], ends + 1))[:-1]
    lengths = np.concatenate((ends, [len(val)]))[:len(starts)] - starts
    starts = starts[lengths >= 4 * args.prompt_tokens]                 # skip very short documents
    picked = []
    for s in starts[np.linspace(0, len(starts) - 1, n).astype(int)]:
        ids = val[s:s + args.prompt_tokens + 1].tolist()
        cut = args.prompt_tokens
        while cut > args.prompt_tokens // 2 and not tok.decode([ids[cut]]).startswith((" ", "\n")):
            cut -= 1                                                    # the next token starts a new word
        picked.append(ids[:cut])
    return picked

@torch.no_grad()
def continue_text(ids, setting, seed):
    t, k, p, r = setting
    torch.manual_seed(seed)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):      # as in sample_stats.py
        idx = model.generate(idx, args.tokens, temperature=t, top_k=int(k) or None, top_p=p or None,
                             repetition_penalty=r or None, stop_token=eot)
    new = idx[0, len(ids):].tolist()
    stopped = eot in new
    text = tok.decode(new[:new.index(eot)] if stopped else new)
    return text + (" [end of text]" if stopped else " [...]")

half = args.n // 2
work = [("tr", ids) for ids in prompts("tr", half)] + [("en", ids) for ids in prompts("en", args.n - half)]
random.Random(args.seed).shuffle(work)                                  # TR and EN mixed in the reading order
order = random.Random(args.seed + 1)
key = {"name": args.name, "first": args.first, "second": args.second, "pairs": []}
lines = [f"{args.n} pairs. After reading both continuations, write A, B or = (no difference) after \"choice:\".",
         "Judge the whole: does it make sense, stay on the prompt's topic, read like real text. Do not open key.json.", ""]
for i, (lang, ids) in enumerate(work, 1):
    texts = {name: continue_text(ids, s, args.seed + i) for name, s in (("first", args.first), ("second", args.second))}
    a, b = ("first", "second") if order.random() < 0.5 else ("second", "first")
    key["pairs"].append({"lang": lang, "A": a, "B": b})
    lines += [f"=== pair {i:02d} " + "=" * 40, "prompt: " + tok.decode(ids), "",
              "A: …" + texts[a], "", "B: …" + texts[b], "", "choice: ", ""]
out.mkdir(parents=True, exist_ok=True)
(out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
(out / "pairs.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out / 'pairs.txt'} ({args.n} pairs: {label(args.first)} vs {label(args.second)}); "
      f"score with: python blind_pairs.py --score --out {args.out}")
