"""Does a Turkish-aware pre-tokenizer help? Two BPE tokenizers trained on the same sample, compared on val text.

The pre-tokenizer cuts text into pieces before BPE runs; BPE never merges across a cut. GPT-2's ByteLevel rule
(what data/bpe_16384.json uses) splits off the English contractions 's 't 're 've 'm 'll 'd, so Turkish suffixes
after an apostrophe are cut in odd places: Türkiye'de -> Türkiye | 'd | e, but Ankara'nın -> Ankara | ' | nın.
A curly apostrophe (’) is not handled at all: Türkçe’nin -> Türkçe | ’ | nin.

    gpt2        the ByteLevel rule, exactly as data_prep.py trains it
    apostrophe  the same rule, but an apostrophe (' or ’) right after a letter or digit stays together with the
                letters that follow it: Türkiye | 'de, Türkçe | ’nin, 2023 | 'te, it | 's, we | 'll

Both train on the same balanced sample of the train shards (data/v2/prep, as data_prep.py) and are measured on
the start of val_tr.bin / val_en.bin, decoded back to text. Nothing in the pipeline changes.

    .venv\\Scripts\\python.exe experiments\\pretok_compare.py            # 300 MB per language, as data_prep.py
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, trainers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data_prep import VOCAB_SIZE, read_shard, shard_names   # the same shard order and vocabulary as the real tokenizer
from textclean import EOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
parser = argparse.ArgumentParser()
parser.add_argument("--mb", type=float, default=300, help="training sample per language, in MB of UTF-8")
parser.add_argument("--eval_tokens", type=int, default=2_000_000, help="val tokens per language to compare on")
parser.add_argument("--data", default="data/v2", help="folder with prep/, val_tr.bin, val_en.bin, bpe_16384.json")
parser.add_argument("--out", default="runs/pretok", help="where the two tokenizers are saved")
args = parser.parse_args()
data, out = ROOT / args.data, ROOT / args.out

# GPT-2's pre-tokenizer regex (what ByteLevel(use_regex=True) runs), and the apostrophe variant
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
APOSTROPHE = r"(?<=[\p{L}\p{N}])['’]\p{L}+| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
# the lookbehind keeps an opening quote ('Hello') out of it: only a suffix right after a word joins

def split_then_bytes(pattern):
    """Cut with our regex on the real characters, then map to bytes without ByteLevel's own regex."""
    return pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])

PRE = {"gpt2": lambda: pre_tokenizers.ByteLevel(add_prefix_space=False),
       "apostrophe": lambda: split_then_bytes(APOSTROPHE)}

# --- the eval text: the start of each val file, decoded with the real tokenizer (byte-level BPE is lossless)
real = Tokenizer.from_file(str(data / json.loads((data / "meta.json").read_text(encoding="utf-8"))["tokenizer"]))
val = {}
for lang in ("tr", "en"):
    ids = np.memmap(data / f"val_{lang}.bin", dtype=np.uint16, mode="r")[:args.eval_tokens].tolist()
    val[lang] = [d for d in real.decode(ids, skip_special_tokens=False).split(EOT) if d]

# control: GPT-2's regex through Split must cut exactly like ByteLevel, or the comparison measures our plumbing
for lang, docs in val.items():
    a, b = PRE["gpt2"](), split_then_bytes(GPT2)
    same = all(a.pre_tokenize_str(d) == b.pre_tokenize_str(d) for d in docs[:2000])
    print(f"control ({lang}, {min(2000, len(docs))} docs): Split(GPT-2 regex) cuts like ByteLevel: {same}")
    if not same:
        sys.exit("the control failed: fix the Split setup before comparing")

# --- train both on the same sample
work = data / "prep"
state = json.loads((work / "state.json").read_text(encoding="utf-8"))

def sample(src):
    n, limit = 0, args.mb * 1e6
    for name in shard_names(state, src):
        for d in read_shard(work, name):
            yield d
            n += len(d.encode("utf-8"))
            if n >= limit:
                return

out.mkdir(parents=True, exist_ok=True)
toks = {}
for name, make in PRE.items():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = make()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t = time.time()
    tok.train_from_iterator((d for src in ("tr", "en") for d in sample(src)), trainer)
    tok.save(str(out / f"{name}.json"))
    toks[name] = tok
    print(f"trained {name}: {tok.get_vocab_size()} tokens in {time.time() - t:.0f} s")

# --- compare
WORD = re.compile(r"\b\w+['’]\w+\b")
print(f"\n{'':<11} | {'TR chars/token':>14} | {'EN chars/token':>14} | {'TR word with apostrophe':>24} | {'EN word with apostrophe':>24}")
for name, tok in toks.items():
    row = []
    for lang in ("tr", "en"):
        docs = val[lang]
        row.append(sum(map(len, docs)) / sum(len(e.ids) for e in tok.encode_batch(docs)))
    for lang in ("tr", "en"):
        words = [" " + w for d in val[lang] for w in WORD.findall(d)]
        row.append(sum(len(e.ids) for e in tok.encode_batch(words)) / len(words))
    print(f"{name:<11} | {row[0]:14.3f} | {row[1]:14.3f} | {row[2]:17.2f} tokens | {row[3]:17.2f} tokens")
print(f"(words with an apostrophe in the val text: TR {sum(len(WORD.findall(d)) for d in val['tr'])}, "
      f"EN {sum(len(WORD.findall(d)) for d in val['en'])})")

for name, tok in toks.items():
    n = sum(1 for t in tok.get_vocab() if t[:1] == "'" or t.startswith("âĢĻ"))   # ’ is âĢĻ in byte-level form
    print(f"{name}: {n} vocabulary entries start with an apostrophe")

print()
for w in ["Türkiye'de", "Ankara'nın", "İstanbul'dan", "Türkçe’nin", "Atatürk'ün", "2023'te", "NATO'ya",
          "it's", "don't", "we'll", "children’s", "'Hello'"]:
    print(f"{w:<14}" + "  ".join(f"{name}: {' | '.join(tok.decode([i]) for i in tok.encode(' ' + w).ids):<32}"
                                 for name, tok in toks.items()))
