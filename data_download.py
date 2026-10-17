import argparse, os, random, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "data" / "mix.txt"
TARGET_BYTES = {"tr": 120_000_000, "en": 80_000_000}   # 200 MB total
MIN_CHARS = 500                                         # drop short stub articles
EOT = "<|endoftext|>"                                   # document separator (special token in the tokenizer)

# lines that start with a year / date, e.g. "1881 - ...", "* MÖ 330", "44 BC"
LIST_LINE = re.compile(r"^\s*[-*•]?\s*(\d{1,4}|M[ÖS]\s*\d+|\d+\s*(BC|AD))\b")
LIST_MIN_LINES = 10
LIST_RATIO = 0.4

# leftovers of templates removed from the dump: "Ankara ( ; , )" or "(; d. 1881)"
EMPTY_PARENS = re.compile(r"\s*\([\s,;:–\-]*\)")
LEADING_PUNCT = re.compile(r"\(\s*[,;][,;\s]*")

# section headings after which a Wikipedia article is only references / link lists
REFERENCE_HEADINGS = {"Kaynakça", "Dış bağlantılar", "Ayrıca bakınız",
                      "References", "External links", "See also"}

def strip_reference_sections(text):
    """Cut the article at the first line that is exactly a reference-section heading."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() in REFERENCE_HEADINGS:
            return "\n".join(lines[:i])
    return text

def strip_category_tail(text):
    """Remove Wikipedia category lines at the end of an article."""
    lines = text.rstrip().splitlines()
    end = len(lines)
    while end > 0:
        s = lines[end - 1].strip()
        if s and len(s) < 80 and not s.endswith((".", "!", "?", ":", '"', ")")):
            end -= 1
        else:
            break
    if len(lines) - end >= 2:
        lines = lines[:end]
    return "\n".join(lines)

def clean(text):
    text = EMPTY_PARENS.sub("", text)        # "( ; , )"      -> ""
    text = LEADING_PUNCT.sub("(", text)      # "(; d. 1881)"  -> "(d. 1881)"
    text = strip_reference_sections(text)    # "Kaynakça", "See also", ... -> cut
    return strip_category_tail(text).strip()

def is_list_page(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < LIST_MIN_LINES:
        return False
    hits = sum(1 for l in lines if LIST_LINE.match(l))
    return hits / len(lines) > LIST_RATIO

def collect(lang, target):
    from datasets import load_dataset        # imported here so clean() can be tested without these packages
    from tqdm import tqdm
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}",
                      split="train", streaming=True)
    docs, total, dropped = [], 0, Counter()
    for article in tqdm(ds, desc=lang):
        text = clean(article["text"])
        if len(text) < MIN_CHARS:
            dropped["short"] += 1
            continue
        if is_list_page(text):
            dropped["list"] += 1
            continue
        docs.append(text)
        total += len(text.encode("utf-8"))
        if total >= target:
            break
    print(f"{lang}: kept {len(docs)} | dropped short {dropped['short']}, list {dropped['list']}")
    return docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download even if mix.txt exists")
    args = parser.parse_args()

    if OUT.exists() and not args.force:
        print(f"{OUT.name} already exists, skipping download (use --force to rebuild)")
        raise SystemExit

    docs = []
    for lang, target in TARGET_BYTES.items():
        docs += collect(lang, target)

    random.seed(42)
    random.shuffle(docs)                     # mix the two languages at document level

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(EOT.join(docs), encoding="utf-8")
    print("done:", round(os.path.getsize(OUT) / 1e6, 1), "MB")
