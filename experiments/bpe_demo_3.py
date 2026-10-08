from collections import Counter
from pathlib import Path
import regex as re, json

ROOT = Path(__file__).parent.parent       # repo root
CORPUS = ROOT / "data" / "mix.txt"
MERGES_FILE = ROOT / "data" / "merges.json"
NUM_MERGES = 1000

# GPT-2 style pre-tokenization: leading space stays attached to the word
PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+""")


def train_merges(text, num_merges):
    """Learn BPE merge rules: repeatedly join the most frequent adjacent pair."""
    freqs = Counter(PATTERN.findall(text))
    words = {w: list(w) for w in freqs}      # every word starts as a list of chars
    merges = []

    for step in range(num_merges):
        # count pairs across the vocabulary, weighted by word frequency
        pairs = Counter()
        for word, count in freqs.items():
            symbols = words[word]
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += count
        if not pairs:
            break

        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))

        for word in words:
            words[word] = apply_merge(words[word], a, b)

        if (step + 1) % 200 == 0:
            total = sum(len(words[w]) * n for w, n in freqs.items())
            print(f"{step + 1:5d} merges | compression {len(text) / total:.2f}")

    return merges


def apply_merge(symbols, a, b):
    """Replace every occurrence of the adjacent pair (a, b) with the single token a+b."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


# --- load cached rules, or train them ---
if MERGES_FILE.exists():
    merges = [tuple(pair) for pair in json.loads(MERGES_FILE.read_text(encoding="utf-8"))]
    print("loaded merges from disk:", len(merges))
else:
    text = CORPUS.read_text(encoding="utf-8")[:200_000]
    merges = train_merges(text, NUM_MERGES)
    MERGES_FILE.write_text(json.dumps(merges, ensure_ascii=False), encoding="utf-8")
    print("saved merges to:", MERGES_FILE)


# --- encoding: replay the merges, in the order they were learned ---
ranks = {pair: i for i, pair in enumerate(merges)}


def tokenize(word):
    """Split any word — including unseen ones — by applying merge rules in learned order."""
    symbols = list(word)
    while len(symbols) > 1:
        # find the applicable rule with the lowest rank (learned earliest)
        rank, pair = min((ranks.get(p, float("inf")), p) for p in zip(symbols, symbols[1:]))
        if rank == float("inf"):
            break                            # no rule applies anymore
        symbols = apply_merge(symbols, *pair)
    return symbols


if __name__ == "__main__":
    print("\n--- words seen during training ---")
    for piece in PATTERN.findall(" Türkiye'de evlerimizden çıkarken gerçekleştirilen çalışmalar"):
        print(f"{piece!r:25} -> {tokenize(piece)}")

    print("\n--- words never seen before ---")
    for unseen in [" umaylaştırmak", " zortlanabilirlik", " tokenizasyon", " Mehmet"]:
        print(f"{unseen!r:25} -> {tokenize(unseen)}")